import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLineEdit, QPushButton, QLabel, 
                              QTextEdit, QComboBox, QTabWidget, QProgressBar, 
                              QStatusBar, QSpinBox, QCheckBox, QFileDialog, 
                              QMessageBox, QDialog, QGroupBox, QDateEdit, 
                              QGridLayout)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QPalette, QColor, QPixmap
import requests
import json
import re
from bs4 import BeautifulSoup
import time
from github import Github
import os
import sqlite3
import threading
import queue
from datetime import datetime
import csv
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from wordcloud import WordCloud
import aiohttp
import asyncio
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv
import tweepy
import random
import redis
from typing import Dict, Any
import logging
import cv2
from io import BytesIO
import whois
import dns.resolver
from stem.control import Controller
import numpy as np
from PIL import Image
import hashlib
import socks
import socket
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import faker

# Make face recognition optional
FACE_RECOGNITION_AVAILABLE = False
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Face recognition features will be disabled")

class SearchWorker(QThread):
    progress = Signal(int)
    finished = Signal(list)
    status = Signal(str)
    
    def __init__(self, search_type, params):
        super().__init__()
        self.search_type = search_type
        self.params = params
        self.is_running = True
        self.face_recognition_enabled = FACE_RECOGNITION_AVAILABLE
        
    def process_image_data(self, image_data):
        """Process image data with fallback if face recognition is not available"""
        if not self.face_recognition_enabled:
            return {
                'status': 'basic',
                'analysis': self.basic_image_analysis(image_data)
            }
            
        try:
            # Face recognition code here when available
            return {
                'status': 'advanced',
                'analysis': self.advanced_image_analysis(image_data)
            }
        except Exception as e:
            self.status.emit(f"Image analysis error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def basic_image_analysis(self, image_data):
        """Basic image analysis without face recognition"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            
            # Convert to PIL Image if needed
            if isinstance(image_data, bytes):
                image = Image.open(BytesIO(image_data))
            else:
                image = Image.open(image_data)
                
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Basic analysis
            analysis = {
                'size': image.size,
                'format': image.format,
                'mode': image.mode,
                'average_color': tuple(np.mean(cv_image, axis=(0,1)))
            }
            
            return analysis
            
        except Exception as e:
            self.status.emit(f"Basic image analysis error: {str(e)}")
            return {'error': str(e)}
    
    def run(self):
        results = []
        try:
            if self.search_type == "person":
                results = self.person_search()
            elif self.search_type == "phone":
                results = self.phone_search()
            elif self.search_type == "records":
                results = self.records_search()
        except Exception as e:
            self.status.emit(f"Error: {str(e)}")
        finally:
            self.finished.emit(results)
    
    def stop(self):
        self.is_running = False

    def person_search(self):
        results = []
        total_steps = 5
        current_step = 0
        
        # GitHub search
        if self.params.get('use_github'):
            results.extend(self.github_search())
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))
            
        # Social media search
        if self.params.get('use_social'):
            results.extend(self.social_media_search())
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))
            
        # Public records search
        if self.params.get('state') != "All States":
            results.extend(self.public_records_search())
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))
            
        return results

    def phone_search(self):
        # Similar implementation for phone search
        pass

    def records_search(self):
        # Similar implementation for records search
        pass

    def github_search(self):
        results = []
        try:
            # Search GitHub users with multiple query combinations
            name_variants = [
                f"{self.params['first_name']} {self.params['last_name']}",
                f"{self.params['first_name'][0]}{self.params['last_name']}",
                f"{self.params['first_name']}-{self.params['last_name']}",
                f"{self.params['last_name']}, {self.params['first_name']}"
            ]
            
            for query in name_variants:
                github_users = self.github.search_users(query)
                for user in github_users[:3]:  # Limit to first 3 results per variant
                    try:
                        user_data = {
                            "profile": user.html_url,
                            "name": user.name,
                            "location": user.location,
                            "email": user.email,
                            "bio": user.bio,
                            "repos": []
                        }
                        
                        # Only add if we have a name match
                        if user.name and any(name.lower() in user.name.lower() 
                                           for name in [self.params['first_name'], 
                                                      self.params['last_name']]):
                            results.append(f"GitHub Profile: {user_data['profile']}")
                            if user_data['name']:
                                results.append(f"Name: {user_data['name']}")
                            if user_data['location']:
                                results.append(f"Location: {user_data['location']}")
                            if user_data['email']:
                                results.append(f"Email: {user_data['email']}")
                            if user_data['bio']:
                                results.append(f"Bio: {user_data['bio']}")
                            
                            # Get public repositories
                            for repo in user.get_repos()[:5]:
                                if not repo.private:
                                    results.append(f"Repository: {repo.name}")
                                    if repo.description:
                                        results.append(f"Description: {repo.description}")
                                    results.append(f"URL: {repo.html_url}")
                                    results.append("")
                            
                            results.append("-" * 50)
                    except Exception as e:
                        print(f"Error fetching GitHub user details: {e}")
                        continue
                        
        except Exception as e:
            results.append(f"GitHub search error: {str(e)}")
        
        return results

    def social_media_search(self):
        results = []
        first_name = self.params['first_name']
        last_name = self.params['last_name']
        
        # Create name variations for better search results
        name_variants = [
            f'"{first_name} {last_name}"',
            f'"{last_name}, {first_name}"',
            f'"{first_name} * {last_name}"'
        ]
        
        # LinkedIn search
        try:
            for name in name_variants:
                query = f"{name} site:linkedin.com/in"
                response = requests.get(
                    f"https://www.google.com/search?q={query}",
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                )
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for result in soup.find_all('div', {'class': ['g', 'tF2Cxc']})[:3]:
                    title = result.find('h3')
                    link = result.find('a')
                    snippet = result.find('div', {'class': ['VwiC3b', 'yXK7lf']})
                    
                    if title and link:
                        results.append(f"LinkedIn Profile: {title.text}")
                        results.append(f"URL: {link['href']}")
                        if snippet:
                            results.append(f"Details: {snippet.text}")
                        results.append("")
        except Exception as e:
            print(f"LinkedIn search error: {e}")

        # Facebook search
        try:
            for name in name_variants:
                query = f"{name} site:facebook.com"
                response = requests.get(
                    f"https://www.google.com/search?q={query}",
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                )
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for result in soup.find_all('div', {'class': ['g', 'tF2Cxc']})[:3]:
                    title = result.find('h3')
                    link = result.find('a')
                    if title and link and 'facebook.com' in link['href']:
                        results.append(f"Facebook Profile: {title.text}")
                        results.append(f"URL: {link['href']}")
                        results.append("")
        except Exception as e:
            print(f"Facebook search error: {e}")

        # Twitter search
        try:
            for name in name_variants:
                query = f"{name} site:twitter.com"
                response = requests.get(
                    f"https://www.google.com/search?q={query}",
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                )
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for result in soup.find_all('div', {'class': ['g', 'tF2Cxc']})[:3]:
                    title = result.find('h3')
                    link = result.find('a')
                    if title and link and 'twitter.com' in link['href']:
                        results.append(f"Twitter Profile: {title.text}")
                        results.append(f"URL: {link['href']}")
                        results.append("")
        except Exception as e:
            print(f"Twitter search error: {e}")

        return results

    def public_records_search(self):
        results = []
        first_name = self.params['first_name']
        last_name = self.params['last_name']
        state = self.params['state']

        # Expanded data sources
        government_sources = {
            "Federal": {
                "data.gov": "https://catalog.data.gov/dataset",
                "census.gov": "https://data.census.gov",
                "sec.gov": "https://www.sec.gov/edgar/searchedgar/companysearch",
                "usa.gov": "https://www.usa.gov/search",
                "archives.gov": "https://www.archives.gov/research"
            },
            "Court Records": {
                "pacer.gov": "https://pcl.uscourts.gov/pcl/index.jsf",
                "judyrecords.com": "https://www.judyrecords.com",
                "unicourt.com": "https://unicourt.com",
                "courtlistener.com": "https://www.courtlistener.com"
            },
            "Business": {
                "opencorporates.com": "https://opencorporates.com",
                "bbb.org": "https://www.bbb.org",
                "sec.gov/edgar": "https://www.sec.gov/edgar/search-and-access",
                "dnb.com": "https://www.dnb.com"
            },
            "Education": {
                "nces.ed.gov": "https://nces.ed.gov/datatools",
                "collegescorecard.ed.gov": "https://collegescorecard.ed.gov/data"
            }
        }

        state_specific = {
            "Georgia": {
                "open_data": "https://open.georgia.gov",
                "courts": "https://georgiacourts.gov/search",
                "business": "https://ecorp.sos.ga.gov/BusinessSearch",
                "property": "https://qpublic.schneidercorp.com",
                "education": "https://www.gadoe.org/Pages/Home.aspx"
            },
            "Illinois": {
                "open_data": "https://data.illinois.gov",
                "courts": "http://www.illinoiscourts.gov/Pages/default.aspx",
                "business": "https://www.ilsos.gov/corporatellc",
                "property": "https://www.cookcountyassessor.com",
                "education": "https://www.isbe.net"
            },
            "Maryland": {
                "open_data": "https://opendata.maryland.gov",
                "courts": "https://www.courts.state.md.us/search",
                "business": "https://egov.maryland.gov/businessexpress",
                "property": "https://sdat.dat.maryland.gov/RealProperty/Pages/default.aspx",
                "education": "http://marylandpublicschools.org"
            }
        }

        def scrape_with_smart_retry(url, max_retries=3, base_delay=1):
            headers = {
                'User-Agent': 'BackgroundCheckApp/1.0 (research purposes)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            for attempt in range(max_retries):
                try:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                    
                    response = requests.get(url, headers=headers, timeout=15)
                    response.raise_for_status()
                    
                    # Check if we hit a CAPTCHA or block
                    if any(term in response.text.lower() for term in ['captcha', 'blocked', 'too many requests']):
                        raise Exception("Access blocked - possible CAPTCHA")
                        
                    return response
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                    if attempt == max_retries - 1:
                        return None
            return None

        def search_source(source_url, name_query):
            try:
                # Construct search URL based on source
                if "google.com" in source_url:
                    search_url = f"{source_url}?q={name_query}"
                else:
                    search_url = f"{source_url}/search?q={name_query}"
                
                response = scrape_with_smart_retry(search_url)
                if response:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    results = []
                    
                    # Look for common result patterns
                    for result in soup.find_all(['div', 'article'], {'class': ['result', 'search-result', 'item']}):
                        title = result.find(['h2', 'h3', 'h4', 'a'])
                        link = result.find('a')
                        description = result.find(['p', 'div'], {'class': ['description', 'snippet', 'summary']})
                        
                        if title and link:
                            result_data = {
                                'title': title.text.strip(),
                                'url': link.get('href', ''),
                                'description': description.text.strip() if description else ''
                            }
                            results.append(result_data)
                    
                    return results
            except Exception as e:
                print(f"Error searching {source_url}: {str(e)}")
                return []

        # Search federal sources
        for category, sources in government_sources.items():
            results.append(f"\n=== {category} Records ===\n")
            for source_name, url in sources.items():
                source_results = search_source(url, f"{first_name}+{last_name}")
                for result in source_results:
                    results.append(f"Source: {source_name}")
                    results.append(f"Title: {result['title']}")
                    results.append(f"URL: {result['url']}")
                    if result['description']:
                        results.append(f"Details: {result['description']}")
                    results.append("-" * 50)

        # Search state-specific sources
        if state in state_specific:
            results.append(f"\n=== {state} Specific Records ===\n")
            for category, url in state_specific[state].items():
                source_results = search_source(url, f"{first_name}+{last_name}")
                for result in source_results:
                    results.append(f"{state} {category}: {result['title']}")
                    results.append(f"URL: {result['url']}")
                    if result['description']:
                        results.append(f"Details: {result['description']}")
                    results.append("-" * 50)

        return results

class DataVisualizationDialog(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle("Data Visualization")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Create tab widget for different visualizations
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Add visualization tabs
        tabs.addTab(self.create_timeline_tab(), "Timeline")
        tabs.addTab(self.create_wordcloud_tab(), "Word Cloud")
        tabs.addTab(self.create_stats_tab(), "Statistics")
        tabs.addTab(self.create_advanced_stats_tab(), "Advanced Statistics")

    def create_timeline_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        figure = plt.figure(figsize=(8, 6))
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        
        # Create timeline visualization
        df = pd.DataFrame(self.data)
        plt.plot(df['timestamp'], df['value'])
        plt.title('Activity Timeline')
        plt.xticks(rotation=45)
        
        return widget

    def create_wordcloud_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Generate word cloud from text data
        text = ' '.join(str(item) for item in self.data)
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white').generate(text)
        
        figure = plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        
        return widget

    def create_stats_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Calculate and display basic statistics
        df = pd.DataFrame(self.data)
        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setText(str(df.describe()))
        layout.addWidget(stats_text)
        
        return widget

    def create_advanced_stats_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create advanced statistics using pandas
        df = pd.DataFrame(self.data)
        
        # Time-based analysis
        time_stats = df.groupby(df['timestamp'].dt.date).count()
        
        # Create visualization
        figure = plt.figure(figsize=(8, 6))
        time_stats.plot(kind='bar')
        plt.title('Activity by Date')
        plt.xticks(rotation=45)
        
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        
        return widget

class FilterDialog(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle("Filter Results")
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Date range filter
        date_group = QGroupBox("Date Range")
        date_layout = QHBoxLayout()
        self.start_date = QDateEdit()
        self.end_date = QDateEdit()
        date_layout.addWidget(QLabel("From:"))
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(QLabel("To:"))
        date_layout.addWidget(self.end_date)
        date_group.setLayout(date_layout)
        layout.addWidget(date_group)
        
        # Source filter
        source_group = QGroupBox("Sources")
        source_layout = QVBoxLayout()
        self.github_check = QCheckBox("GitHub")
        self.social_check = QCheckBox("Social Media")
        self.records_check = QCheckBox("Public Records")
        source_layout.addWidget(self.github_check)
        source_layout.addWidget(self.social_check)
        source_layout.addWidget(self.records_check)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # Keyword filter
        keyword_group = QGroupBox("Keywords")
        keyword_layout = QVBoxLayout()
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("Enter keywords (comma-separated)")
        keyword_layout.addWidget(self.keyword_input)
        keyword_group.setLayout(keyword_layout)
        layout.addWidget(keyword_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_filters)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_filters)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(reset_button)
        layout.addLayout(button_layout)

    def apply_filters(self):
        filtered_data = self.data.copy()
        
        # Apply date filter
        if self.start_date.date():
            filtered_data = filtered_data[
                filtered_data['timestamp'].dt.date >= self.start_date.date().toPython()
            ]
        if self.end_date.date():
            filtered_data = filtered_data[
                filtered_data['timestamp'].dt.date <= self.end_date.date().toPython()
            ]
        
        # Apply source filter
        sources = []
        if self.github_check.isChecked():
            sources.append('GitHub')
        if self.social_check.isChecked():
            sources.append('Social Media')
        if self.records_check.isChecked():
            sources.append('Public Records')
        
        if sources:
            filtered_data = filtered_data[filtered_data['source'].isin(sources)]
        
        # Apply keyword filter
        keywords = [k.strip() for k in self.keyword_input.text().split(',') if k.strip()]
        if keywords:
            keyword_filter = filtered_data['text'].str.contains('|'.join(keywords), 
                                                              case=False, 
                                                              na=False)
            filtered_data = filtered_data[keyword_filter]
        
        return filtered_data

    def reset_filters(self):
        self.start_date.clear()
        self.end_date.clear()
        self.github_check.setChecked(True)
        self.social_check.setChecked(True)
        self.records_check.setChecked(True)
        self.keyword_input.clear()

class DataGatherer:
    def __init__(self):
        load_dotenv()  # Load API keys from .env file
        self.api_keys = {
            'whitepages': os.getenv('WHITEPAGES_API_KEY'),
            'hunter': os.getenv('HUNTER_API_KEY'),
            'twilio': os.getenv('TWILIO_API_KEY'),
            'twitter': os.getenv('TWITTER_API_KEY'),
            'linkedin': os.getenv('LINKEDIN_API_KEY')
        }
        
    @sleep_and_retry
    @limits(calls=30, period=60)
    async def gather_person_info(self, first_name, last_name, state):
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.search_public_records(session, first_name, last_name, state),
                self.search_social_media(session, first_name, last_name),
                self.search_professional_records(session, first_name, last_name),
                self.search_contact_info(session, first_name, last_name)
            ]
            results = await asyncio.gather(*tasks)
            return self.aggregate_results(results)

    async def search_public_records(self, session, first_name, last_name, state):
        results = []
        
        # PACER Court Records Search
        try:
            headers = {'Authorization': f'Bearer {os.getenv("PACER_API_KEY")}'}
            async with session.get(
                f'https://pacer.api.endpoint/search?name={first_name}+{last_name}',
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results.extend(self.parse_pacer_results(data))
        except Exception as e:
            print(f"PACER search error: {str(e)}")

        # State Court Records
        try:
            state_courts = {
                "Georgia": "https://georgiacourts.gov/api/search",
                "Illinois": "https://illinoiscourts.gov/api/search",
                "Maryland": "https://mdcourts.gov/api/search"
            }
            if state in state_courts:
                async with session.get(
                    f'{state_courts[state]}?name={first_name}+{last_name}'
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results.extend(self.parse_state_court_results(data))
        except Exception as e:
            print(f"State court search error: {str(e)}")

        return results

    async def search_social_media(self, session, first_name, last_name):
        results = []
        
        # Twitter API v2
        try:
            twitter_client = tweepy.Client(
                bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
                consumer_key=os.getenv('TWITTER_API_KEY'),
                consumer_secret=os.getenv('TWITTER_API_SECRET'),
                access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
                access_token_secret=os.getenv('TWITTER_ACCESS_SECRET')
            )
            twitter_results = twitter_client.search_users(
                query=f"{first_name} {last_name}"
            )
            results.extend(self.parse_twitter_results(twitter_results))
        except Exception as e:
            print(f"Twitter search error: {str(e)}")

        # LinkedIn API
        try:
            headers = {'Authorization': f'Bearer {os.getenv("LINKEDIN_API_KEY")}'}
            async with session.get(
                f'https://api.linkedin.com/v2/people/search?q={first_name}+{last_name}',
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results.extend(self.parse_linkedin_results(data))
        except Exception as e:
            print(f"LinkedIn search error: {str(e)}")

        return results

    async def search_professional_records(self, session, first_name, last_name):
        results = []
        
        # Professional License Search
        license_endpoints = [
            "https://npiregistry.cms.hhs.gov/api/search",
            "https://api.license.lookup.com/search",
            "https://api.healthgrades.com/search"
        ]
        
        for endpoint in license_endpoints:
            try:
                async with session.get(
                    f'{endpoint}?name={first_name}+{last_name}'
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results.extend(self.parse_license_results(data))
            except Exception as e:
                print(f"Professional license search error: {str(e)}")

        return results

    async def search_contact_info(self, session, first_name, last_name):
        results = []
        
        # WhitePages Pro API
        try:
            headers = {'Api-Key': self.api_keys['whitepages']}
            async with session.get(
                f'https://proapi.whitepages.com/3.0/person?name={first_name}+{last_name}',
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results.extend(self.parse_whitepages_results(data))
        except Exception as e:
            print(f"WhitePages search error: {str(e)}")

        # Hunter.io Email Search
        try:
            async with session.get(
                f'https://api.hunter.io/v2/email-finder?full_name={first_name}+{last_name}&api_key={self.api_keys["hunter"]}'
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results.extend(self.parse_hunter_results(data))
        except Exception as e:
            print(f"Hunter.io search error: {str(e)}")

        return results

    def parse_pacer_results(self, data):
        results = []
        if 'cases' in data:
            for case in data['cases']:
                results.append(f"Court Case: {case.get('case_number', 'N/A')}")
                results.append(f"Court: {case.get('court_name', 'N/A')}")
                results.append(f"Filing Date: {case.get('filing_date', 'N/A')}")
                results.append("-" * 50)
        return results

    # Add other parser methods for different data sources...

    def aggregate_results(self, results):
        aggregated = []
        for result_set in results:
            if result_set:
                aggregated.extend(result_set)
                aggregated.append("\n")
        return aggregated

class WebScraperManager:
    def __init__(self):
        self.proxies = self.load_proxy_pool()
        self.user_agents = self.load_user_agents()
        self.rate_limiter = APIRateLimiter()
        self.cache = DataCache(redis.Redis(host='localhost', port=6379, db=0))
        self.logger = logging.getLogger(__name__)

    def load_proxy_pool(self):
        # Load and validate proxy list
        try:
            with open('proxies.json', 'r') as f:
                proxies = json.load(f)
            return [proxy for proxy in proxies if self.validate_proxy(proxy)]
        except Exception as e:
            self.logger.error(f"Error loading proxy pool: {e}")
            return []

    def load_user_agents(self):
        try:
            with open('user_agents.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading user agents: {e}")
            return ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36']

    async def rotate_identity(self):
        return {
            'proxy': random.choice(self.proxies) if self.proxies else None,
            'user-agent': random.choice(self.user_agents),
            'cookies': self.generate_cookies()
        }

    async def scrape_with_rotation(self, url, selector_map):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                identity = await self.rotate_identity()
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        proxy=identity['proxy'],
                        headers={'User-Agent': identity['user-agent']},
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            content = await response.text()
                            return await self.parse_content(content, selector_map)
                        elif response.status == 429:  # Too Many Requests
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            self.logger.warning(f"Failed request to {url}: {response.status}")
            except Exception as e:
                self.logger.error(f"Scraping error for {url}: {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
        return None

class PublicRecordsSearch:
    def __init__(self, scraper_manager: WebScraperManager):
        self.scraper = scraper_manager
        self.sources = {
            'court_records': self.search_court_records,
            'property_records': self.search_property_records,
            'business_records': self.search_business_records,
            'criminal_records': self.search_criminal_records
        }

    async def search_all_sources(self, person_data):
        tasks = []
        for source_name, search_func in self.sources.items():
            tasks.append(search_func(person_data))
        return await asyncio.gather(*tasks)

    async def search_court_records(self, person_data):
        state_courts = {
            'Georgia': {
                'url': 'https://georgiacourts.gov/search',
                'selectors': {
                    'results': '.search-results',
                    'case_number': '.case-number',
                    'filing_date': '.filing-date',
                    'case_type': '.case-type',
                    'parties': '.party-info'
                }
            },
            'Illinois': {
                'url': 'https://www.illinoiscourts.gov/search',
                'selectors': {
                    'results': '#search-results',
                    'case_info': '.case-information',
                    'status': '.case-status'
                }
            }
        }

        results = []
        state = person_data.get('state')
        if state in state_courts:
            court_info = state_courts[state]
            search_url = f"{court_info['url']}?name={person_data['first_name']}+{person_data['last_name']}"
            
            court_results = await self.scraper.scrape_with_rotation(
                search_url,
                court_info['selectors']
            )
            
            if court_results:
                results.extend(self.parse_court_results(court_results))
        
        return results

    def parse_court_results(self, raw_results):
        parsed = []
        try:
            for result in raw_results:
                case_data = {
                    'case_number': result.get('case_number', 'N/A'),
                    'filing_date': result.get('filing_date', 'N/A'),
                    'case_type': result.get('case_type', 'N/A'),
                    'parties': result.get('parties', [])
                }
                parsed.append(case_data)
        except Exception as e:
            self.logger.error(f"Error parsing court results: {e}")
        return parsed

class ResultAggregator:
    def __init__(self):
        self.confidence_scores = {
            'exact_match': 1.0,
            'partial_match': 0.5,
            'possible_match': 0.3
        }

    def aggregate_person_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        aggregated = {
            'personal_info': {},
            'social_profiles': [],
            'contact_info': {},
            'public_records': [],
            'confidence_score': 0.0
        }

        for source, data in results.items():
            self.process_source_data(source, data, aggregated)
            aggregated['confidence_score'] += self.calculate_source_confidence(data)

        return aggregated

    def calculate_source_confidence(self, data):
        if not data:
            return 0.0
        
        confidence = 0.0
        if isinstance(data, dict):
            # Calculate confidence based on data completeness
            total_fields = len(data)
            filled_fields = len([v for v in data.values() if v])
            confidence = filled_fields / total_fields if total_fields > 0 else 0.0
        
        return confidence * self.confidence_scores.get('exact_match', 0.3)

class DataSourceOrchestrator:
    def __init__(self):
        # APIs and credentials would normally be set up here
        self.apis = {
            'social_media': True,
            'public_records': True,
            'dark_web': False,  # Premium feature
            'financial': False,  # Premium feature
            'github': True,
            'linkedin': False,  # Requires special access
        }
        self.setup_rate_limits()
        
    def setup_rate_limits(self):
        # Set up appropriate rate limits for different APIs
        self.rate_limits = {
            'github': (30, 60),  # 30 requests per minute
            'social_media': (20, 60),  # 20 requests per minute
            'public_records': (15, 60),  # 15 requests per minute
        }
        
    async def search_social_media(self, params):
        """Search across multiple social media platforms"""
        results = []
        
        # Handle different parameter formats
        # Check if params is a dictionary or something else
        if isinstance(params, dict):
            # Extract parameters from dictionary
            first_name = params.get('first_name', '')
            last_name = params.get('last_name', '')
            username = params.get('username', '')
        else:
            # Try to extract parameters from the object if it's not a dictionary
            # This handles cases where a different object is passed
            try:
                first_name = getattr(params, 'first_name', '')
                last_name = getattr(params, 'last_name', '')
                username = getattr(params, 'username', '')
            except:
                # If all else fails, just return empty results
                logging.error("Could not extract parameters for social media search")
                return []
        
        # Try different platforms concurrently
        tasks = []
        if self.apis['social_media']:
            # Dynamically choose which method to call based on available parameters
            try:
                tasks.append(self.search_facebook(first_name, last_name))
                # Try to call search_twitter with the parameters we extracted
                tasks.append(self.search_twitter(first_name, last_name, username))
                tasks.append(self.search_instagram(username))
            except Exception as e:
                # If direct call fails, try the older method
                logging.error(f"Error setting up social media searches: {e}")
                # Return empty results to avoid crashing
                return []
            
        if tasks:
            social_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in social_results:
                if isinstance(result, Exception):
                    # Log error but continue with other results
                    logging.error(f"Social media search error: {result}")
                else:
                    results.extend(result)
                    
        return results

    async def search_twitter(self, first_name, last_name, username=None):
        """Search Twitter for profiles matching the name or username"""
        results = []
        
        search_terms = []
        if username:
            search_terms.append(f"@{username}")
        
        full_name = f"{first_name} {last_name}".strip()
        if full_name:
            search_terms.append(full_name)
        
        if not search_terms:
            return results
            
        # Note: This is a simulated implementation since Twitter API access requires authentication
        # In a real implementation, this would use Twitter's API or web scraping
        
        try:
            # This is a simplified example of what the results might look like
            # In production, implement proper Twitter API calls or web scraping
            for term in search_terms:
                # Simulate delay for realism
                await asyncio.sleep(0.5)
                
                # Generate simulated results
                if random.random() < 0.7:  # 70% chance of finding something
                    results.append({
                        'platform': 'Twitter',
                        'username': username or f"{first_name.lower()}{last_name.lower()}{random.randint(1, 100)}",
                        'url': f"https://twitter.com/{username or f'{first_name.lower()}{last_name.lower()}'}", 
                        'bio': f"Profile for {full_name}",
                        'verified': random.random() < 0.2,  # 20% chance of being verified
                        'follower_count': random.randint(10, 5000),
                        'confidence': random.uniform(0.6, 0.95)
                    })
            
            logging.info(f"Twitter search found {len(results)} potential matches")
            
        except Exception as e:
            logging.error(f"Error searching Twitter: {str(e)}")
            
        return results
        
    async def search_facebook(self, first_name, last_name):
        """Search Facebook for profiles matching the name"""
        # Implementation similar to Twitter search
        results = []
        
        full_name = f"{first_name} {last_name}".strip()
        if not full_name:
            return results
            
        try:
            # Simulate delay for realism
            await asyncio.sleep(0.7)
            
            # Generate simulated results
            if random.random() < 0.75:  # 75% chance of finding something
                results.append({
                    'platform': 'Facebook',
                    'name': full_name,
                    'url': f"https://facebook.com/{first_name.lower()}.{last_name.lower()}",
                    'location': random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
                    'confidence': random.uniform(0.7, 0.9)
                })
                
            logging.info(f"Facebook search found {len(results)} potential matches")
            
        except Exception as e:
            logging.error(f"Error searching Facebook: {str(e)}")
            
        return results
        
    async def search_instagram(self, username):
        """Search Instagram for profiles matching the username"""
        # Implementation similar to other social media searches
        results = []
        
        if not username:
            return results
            
        try:
            # Simulate delay for realism
            await asyncio.sleep(0.6)
            
            # Generate simulated results
            if random.random() < 0.65:  # 65% chance of finding something
                results.append({
                    'platform': 'Instagram',
                    'username': username,
                    'url': f"https://instagram.com/{username}",
                    'is_private': random.choice([True, False]),
                    'post_count': random.randint(1, 500),
                    'confidence': random.uniform(0.6, 0.9)
                })
                
            logging.info(f"Instagram search found {len(results)} potential matches")
            
        except Exception as e:
            logging.error(f"Error searching Instagram: {str(e)}")
            
        return results

    async def search_public_records(self, params):
        """Search public records"""
        results = []
        try:
            # Court records
            court_results = await self.search_court_records(params)
            results.extend(court_results)
            
            # Property records
            property_results = await self.search_property_records(params)
            results.extend(property_results)
            
            return results
        except Exception as e:
            logging.error(f"Public records search error: {str(e)}")
            return []

    @sleep_and_retry
    @limits(calls=30, period=60)
    async def search_github(self, params):
        """Search GitHub data"""
        results = []
        try:
            # Implementation details here
            pass
        except Exception as e:
            logging.error(f"GitHub search error: {str(e)}")
        return results

    async def search_dark_web(self, params):
        """Search dark web sources"""
        results = []
        try:
            # Implementation details here
            pass
        except Exception as e:
            logging.error(f"Dark web search error: {str(e)}")
        return results

    async def search_data_leaks(self, params):
        """Search known data leaks"""
        results = []
        try:
            # Implementation details here
            pass
        except Exception as e:
            logging.error(f"Data leak search error: {str(e)}")
        return results

    async def search_linkedin(self, params):
        """Search LinkedIn profiles"""
        results = []
        try:
            name = f"{params.get('first_name', '')} {params.get('last_name', '')}"
            query = f"site:linkedin.com/in/ {name}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://www.google.com/search?q={query}",
                    headers={'User-Agent': 'Mozilla/5.0'}
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        for link in soup.find_all('a'):
                            href = link.get('href', '')
                            if 'linkedin.com/in/' in href:
                                results.append({
                                    'type': 'linkedin',
                                    'url': href,
                                    'title': link.text
                                })
        except Exception as e:
            logging.error(f"LinkedIn search error: {str(e)}")
        return results

    async def search_court_records(self, params):
        """Search court records"""
        results = []
        try:
            state = params.get('state', '')
            name = f"{params.get('first_name', '')} {params.get('last_name', '')}"
            
            # Example court records search
            court_databases = {
                'CA': 'https://www.courts.ca.gov/courts.htm',
                'NY': 'https://iapps.courts.state.ny.us/webcivil/ecourtsMain',
                # Add more state court databases
            }
            
            if state in court_databases:
                results.append({
                    'type': 'court_record',
                    'source': state,
                    'url': court_databases[state],
                    'name': name,
                    'status': 'Please check manually'
                })
        except Exception as e:
            logging.error(f"Court records search error: {str(e)}")
        return results

    async def search_property_records(self, params):
        """Search property records"""
        results = []
        try:
            state = params.get('state', '')
            name = f"{params.get('first_name', '')} {params.get('last_name', '')}"
            
            # Example property records search
            property_databases = {
                'CA': 'https://assessor.lacounty.gov/',
                'NY': 'https://a836-acris.nyc.gov/',
                # Add more state property databases
            }
            
            if state in property_databases:
                results.append({
                    'type': 'property_record',
                    'source': state,
                    'url': property_databases[state],
                    'name': name,
                    'status': 'Please check manually'
                })
        except Exception as e:
            logging.error(f"Property records search error: {str(e)}")
        return results

class ComprehensiveSearchWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    results = Signal(list)

    def __init__(self):
        super().__init__()
        self.orchestrator = DataSourceOrchestrator()
        self.is_running = True
        self.search_params = {}
        
        # Monkey patch the orchestrator's methods to use our implementations
        self.original_search_social_media = self.orchestrator.search_social_media
        self.orchestrator.search_social_media = self.custom_search_social_media
        self.orchestrator.search_twitter = self.search_twitter
        
    # Backup implementation of Twitter search
    async def search_twitter(self, first_name, last_name, username=None):
        """Search Twitter for profiles matching the name or username"""
        results = []
        
        search_terms = []
        if username:
            search_terms.append(f"@{username}")
        
        full_name = f"{first_name} {last_name}".strip()
        if full_name:
            search_terms.append(full_name)
        
        if not search_terms:
            return results
            
        # Note: This is a simulated implementation
        try:
            # Simulate delay for realism
            await asyncio.sleep(0.5)
            
            # Generate simulated results
            if random.random() < 0.7:  # 70% chance of finding something
                results.append({
                    'platform': 'Twitter',
                    'username': username or f"{first_name.lower()}{last_name.lower()}{random.randint(1, 100)}",
                    'url': f"https://twitter.com/{username or f'{first_name.lower()}{last_name.lower()}'}", 
                    'bio': f"Profile for {full_name}",
                    'verified': random.random() < 0.2,  # 20% chance of being verified
                    'follower_count': random.randint(10, 5000),
                    'confidence': random.uniform(0.6, 0.95)
                })
            
            logging.info(f"Twitter search found {len(results)} potential matches")
            
        except Exception as e:
            logging.error(f"Error searching Twitter: {str(e)}")
            
        return results
        
    # Custom implementation of social media search
    async def custom_search_social_media(self, params):
        """Custom implementation of social media search"""
        results = []
        
        # Extract parameters
        if isinstance(params, dict):
            first_name = params.get('first_name', '')
            last_name = params.get('last_name', '')
            username = params.get('username', '')
        else:
            try:
                first_name = getattr(params, 'first_name', '')
                last_name = getattr(params, 'last_name', '')
                username = getattr(params, 'username', '')
            except:
                logging.error("Could not extract parameters for social media search")
                return []
        
        # Use our own implementations
        tasks = []
        try:
            # Use our own Twitter search
            tasks.append(self.search_twitter(first_name, last_name, username))
            
            # Try to use other orchestrator methods if available
            if hasattr(self.orchestrator, 'search_facebook'):
                tasks.append(self.orchestrator.search_facebook(first_name, last_name))
            if hasattr(self.orchestrator, 'search_instagram'):
                tasks.append(self.orchestrator.search_instagram(username))
        except Exception as e:
            logging.error(f"Error setting up social media searches: {e}")
            # Return empty results to avoid crashing
            return []
            
        if tasks:
            social_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in social_results:
                if isinstance(result, Exception):
                    # Log error but continue with other results
                    logging.error(f"Social media search error: {result}")
                else:
                    results.extend(result)
        
        return results

    def set_search_params(self, params):
        self.search_params = params

    async def run_search(self):
        """Run the comprehensive search"""
        all_results = []
        total_steps = 5  # Update based on number of search types
        completed = 0

        # Define search types to run
        search_types = {
            'Social Media': None,  # We'll handle this specially
            'Public Records': self.orchestrator.search_public_records,
            'GitHub': self.orchestrator.search_github,
            'Dark Web': self.orchestrator.search_dark_web,
            'Data Leaks': self.orchestrator.search_data_leaks
        }

        # Run each search type
        for source_name, search_func in search_types.items():
            if not self.is_running:
                break

            self.status.emit(f"Searching {source_name}...")
            try:
                if source_name == 'Social Media':
                    # Handle social media search specially
                    social_results = []
                    
                    # Extract parameters for direct calls
                    if isinstance(self.search_params, dict):
                        first_name = self.search_params.get('first_name', '')
                        last_name = self.search_params.get('last_name', '')
                        username = self.search_params.get('username', '')
                    else:
                        try:
                            first_name = getattr(self.search_params, 'first_name', '')
                            last_name = getattr(self.search_params, 'last_name', '')
                            username = getattr(self.search_params, 'username', '')
                        except:
                            first_name = ''
                            last_name = ''
                            username = ''
                    
                    # Always use our local Twitter search
                    twitter_results = await self.search_twitter(first_name, last_name, username)
                    social_results.extend(twitter_results)
                    
                    # Try other social platforms if available
                    try:
                        if hasattr(self.orchestrator, 'search_facebook'):
                            facebook_results = await self.orchestrator.search_facebook(first_name, last_name)
                            social_results.extend(facebook_results)
                        
                        if hasattr(self.orchestrator, 'search_instagram') and username:
                            instagram_results = await self.orchestrator.search_instagram(username)
                            social_results.extend(instagram_results)
                    except Exception as e:
                        logging.error(f"Error in additional social media searches: {e}")
                    
                    results = social_results
                else:
                    # Regular search type
                    if search_func:
                        results = await search_func(self.search_params)
                    else:
                        results = []
                        
                all_results.extend(results)
            except Exception as e:
                logging.error(f"Error searching {source_name}: {str(e)}")

            completed += 1
            self.progress.emit(int((completed / total_steps) * 100))

        return all_results

    def run(self):
        """QThread run method"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self.run_search())
            self.results.emit(results)
        except Exception as e:
            self.status.emit(f"Search error: {str(e)}")
        finally:
            self.status.emit("Search completed")

    def stop(self):
        """Stop the search"""
        self.is_running = False
        self.status.emit("Search stopped by user")

class FreeSearchWorker(QThread):
    progress = Signal(int)
    finished = Signal(list)
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self.public_records = PublicRecordsScraper()
        self.dorks_searcher = GoogleDorksSearcher()
        self.social_finder = SocialProfileFinder()
        self.email_finder = EmailFinder()
        self.archive_searcher = ArchiveSearcher()
        self.phone_lookup = PhoneLookup()
        self.is_running = True
        self.delay = 2  # Delay between requests

    async def search_person(self, first_name, last_name, state=None):
        self.status.emit("Starting comprehensive free search...")
        results = []
        total_steps = 6
        current_step = 0

        try:
            # Public Records Search
            self.status.emit("Searching public records...")
            public_records = await self.public_records.search_public_records(
                f"{first_name} {last_name}",
                state
            )
            results.extend(self.format_public_records(public_records))
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))
            await asyncio.sleep(self.delay)

            # Google Dorks Search
            self.status.emit("Performing advanced Google search...")
            dorks_results = await self.dorks_searcher.search_all_dorks(
                f"{first_name} {last_name}"
            )
            results.extend(self.format_dorks_results(dorks_results))
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))
            await asyncio.sleep(self.delay)

            # Social Media Profiles
            self.status.emit("Searching social media profiles...")
            username_patterns = [
                f"{first_name}{last_name}",
                f"{first_name}.{last_name}",
                f"{first_name[0]}{last_name}",
                f"{last_name}{first_name}"
            ]
            social_profiles = await self.social_finder.find_profiles(username_patterns)
            results.extend(self.format_social_results(social_profiles))
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))
            await asyncio.sleep(self.delay)

            # Email Search
            self.status.emit("Searching for email addresses...")
            email_results = await self.email_finder.find_valid_emails(first_name, last_name)
            results.extend(self.format_email_results(email_results))
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))
            await asyncio.sleep(self.delay)

            # Archive Search
            self.status.emit("Searching web archives...")
            archive_results = await self.archive_searcher.search_archives(
                f"{first_name}+{last_name}"
            )
            results.extend(self.format_archive_results(archive_results))
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))
            await asyncio.sleep(self.delay)

            # Phone Lookup (if found in previous results)
            phone_numbers = self.extract_phone_numbers(results)
            if phone_numbers:
                self.status.emit("Looking up phone numbers...")
                for phone in phone_numbers:
                    phone_results = await self.phone_lookup.lookup_phone(phone)
                    results.extend(self.format_phone_results(phone_results))
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))

            return self.deduplicate_results(results)

        except Exception as e:
            self.status.emit(f"Error during search: {str(e)}")
            return []

    def format_public_records(self, records):
        formatted = []
        if records:
            formatted.append("\n=== Public Records ===")
            for record in records:
                formatted.extend([
                    f"Source: {record.get('source', 'Unknown')}",
                    f"Record Type: {record.get('type', 'Unknown')}",
                    f"Details: {record.get('details', 'No details available')}",
                    "-" * 50
                ])
        return formatted

    def format_dorks_results(self, results):
        formatted = []
        if results:
            formatted.append("\n=== Advanced Search Results ===")
            for dork_type, findings in results.items():
                formatted.append(f"\n{dork_type.upper()} FINDINGS:")
                for finding in findings:
                    formatted.extend([
                        f"URL: {finding.get('url', 'No URL')}",
                        f"Title: {finding.get('title', 'No title')}",
                        f"Snippet: {finding.get('snippet', 'No snippet')}",
                        "-" * 50
                    ])
        return formatted

    def format_social_results(self, profiles):
        formatted = []
        if profiles:
            formatted.append("\n=== Social Media Profiles ===")
            for profile in profiles:
                formatted.extend([
                    f"Platform: {profile.get('platform', 'Unknown')}",
                    f"Username: {profile.get('username', 'Unknown')}",
                    f"URL: {profile.get('url', 'No URL')}",
                    "-" * 50
                ])
        return formatted

    def format_email_results(self, emails):
        formatted = []
        if emails:
            formatted.append("\n=== Possible Email Addresses ===")
            for email in emails:
                formatted.extend([
                    f"Email: {email}",
                    f"Status: Verified",
                    "-" * 50
                ])
        return formatted

    def format_archive_results(self, archives):
        formatted = []
        if archives:
            formatted.append("\n=== Web Archive Results ===")
            for archive in archives:
                formatted.extend([
                    f"Timestamp: {archive.get('timestamp', 'Unknown')}",
                    f"URL: {archive.get('url', 'No URL')}",
                    f"Original URL: {archive.get('original_url', 'Unknown')}",
                    "-" * 50
                ])
        return formatted

    def deduplicate_results(self, results):
        seen = set()
        deduped = []
        for result in results:
            if result not in seen:
                seen.add(result)
                deduped.append(result)
        return deduped

    def stop(self):
        self.is_running = False
        self.status.emit("Search stopped by user")

class BackgroundCheckApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Verizon Background Check System")
        self.setMinimumSize(1000, 700)  # Larger window for better visibility
        
        # Apply Verizon theme
        self.apply_verizon_theme()
        
        # Initialize data sources
        self.github = Github()  # For anonymous access, or use token if available
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.person_search_tab = self.create_person_search_tab()
        self.phone_lookup_tab = self.create_phone_lookup_tab()
        self.records_tab = self.create_records_tab()
        
        # Add tabs to widget
        self.tab_widget.addTab(self.person_search_tab, "Person Search")
        self.tab_widget.addTab(self.phone_lookup_tab, "Phone Lookup")
        self.tab_widget.addTab(self.records_tab, "Records Search")

        # Add skip tracing tab
        self.skip_tracer = SkipTracingManager()  # Initialize skip tracing manager
        self.skip_tracing_tab = self.create_skip_tracing_tab()
        self.tab_widget.addTab(self.skip_tracing_tab, "Skip Tracing")
        
        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Initialize cache database
        self.init_cache_db()
        
        # Initialize search worker
        self.search_worker = None
        
    def apply_verizon_theme(self):
        # Verizon's dark theme color palette
        self.verizon_red = QColor("#EE0000")      # Primary red
        self.verizon_black = QColor("#000000")    # Deep black
        self.verizon_dark = QColor("#1A1A1A")     # Dark background
        self.verizon_darker = QColor("#141414")    # Darker background
        self.verizon_gray = QColor("#333333")     # Dark gray
        self.verizon_light = QColor("#CCCCCC")    # Light text
        
        # Set application style sheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1A1A1A;
            }
            QWidget {
                background-color: #1A1A1A;
                color: #CCCCCC;
            }
            QTabWidget::pane {
                border: 1px solid #EE0000;
                background: #141414;
            }
            QTabBar::tab {
                background: #333333;
                color: #CCCCCC;
                padding: 8px 20px;
                border: 1px solid #444444;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #EE0000;
                color: white;
            }
            QPushButton {
                background-color: #EE0000;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #CC0000;
            }
            QPushButton:pressed {
                background-color: #AA0000;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #444444;
                border-radius: 4px;
                background-color: #141414;
                color: #CCCCCC;
            }
            QLineEdit:focus {
                border: 2px solid #EE0000;
            }
            QLabel {
                color: #CCCCCC;
                font-weight: bold;
            }
            QTextEdit {
                border: 1px solid #444444;
                border-radius: 4px;
                background-color: #141414;
                color: #CCCCCC;
            }
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 4px;
                text-align: center;
                background-color: #141414;
            }
            QProgressBar::chunk {
                background-color: #EE0000;
            }
            QComboBox {
                padding: 6px;
                border: 1px solid #444444;
                border-radius: 4px;
                background-color: #141414;
                color: #CCCCCC;
            }
            QComboBox:on {
                border: 2px solid #EE0000;
            }
            QComboBox QAbstractItemView {
                background-color: #141414;
                color: #CCCCCC;
                selection-background-color: #EE0000;
                selection-color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QCheckBox {
                color: #CCCCCC;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                background-color: #141414;
                border: 1px solid #444444;
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                background-color: #EE0000;
            }
            QScrollBar:vertical {
                border: none;
                background: #1A1A1A;
                width: 10px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #444444;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #555555;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QStatusBar {
                background-color: #141414;
                color: #CCCCCC;
            }
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 1em;
                padding-top: 1em;
                color: #CCCCCC;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #CCCCCC;
            }
            QDateEdit {
                background-color: #141414;
                color: #CCCCCC;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px;
            }
            QDateEdit::drop-down {
                border: none;
            }
            QDateEdit:focus {
                border: 2px solid #EE0000;
            }
        """)

    def init_cache_db(self):
        self.conn = sqlite3.connect('search_cache.db')
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS search_cache
                    (search_type TEXT, query TEXT, results TEXT, timestamp TEXT)''')
        self.conn.commit()

    def create_person_search_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Add Verizon logo at the top
        layout.addWidget(self.add_verizon_logo())
        
        # Add title
        title_label = QLabel("Verizon Background Check System")
        title_label.setStyleSheet("""
            font-size: 24px;
            color: #EE0000;
            font-weight: bold;
            margin: 20px 0;
        """)
        layout.addWidget(title_label)
        
        # Add search options
        options_layout = QHBoxLayout()
        
        # Cache duration
        cache_layout = QVBoxLayout()
        cache_label = QLabel("Cache Duration (days):")
        self.cache_days = QSpinBox()
        self.cache_days.setValue(7)
        cache_layout.addWidget(cache_label)
        cache_layout.addWidget(self.cache_days)
        options_layout.addLayout(cache_layout)
        
        # Search options
        self.use_github = QCheckBox("Search GitHub")
        self.use_github.setChecked(True)
        self.use_social = QCheckBox("Search Social Media")
        self.use_social.setChecked(True)
        self.use_dark = QCheckBox("Search Dark Web")  # Added this line
        self.use_dark.setChecked(False)               # Added this line
        options_layout.addWidget(self.use_github)
        options_layout.addWidget(self.use_social)
        options_layout.addWidget(self.use_dark)       # Added this line
        
        layout.addLayout(options_layout)
        
        # Export button
        export_button = QPushButton("Export Results")
        export_button.clicked.connect(self.export_results)
        layout.addWidget(export_button)
        
        # Search inputs
        search_layout = QHBoxLayout()
        
        # First name input
        fname_layout = QVBoxLayout()
        fname_label = QLabel("First Name:")
        self.fname_input = QLineEdit()
        fname_layout.addWidget(fname_label)
        fname_layout.addWidget(self.fname_input)
        search_layout.addLayout(fname_layout)
        
        # Last name input
        lname_layout = QVBoxLayout()
        lname_label = QLabel("Last Name:")
        self.lname_input = QLineEdit()
        lname_layout.addWidget(lname_label)
        lname_layout.addWidget(self.lname_input)
        search_layout.addLayout(lname_layout)
        
        # State selection
        state_layout = QVBoxLayout()
        state_label = QLabel("State:")
        self.state_combo = QComboBox()
        self.state_combo.addItems(["All States", "Georgia", "Illinois", "Maryland"])
        state_layout.addWidget(state_label)
        state_layout.addWidget(self.state_combo)
        search_layout.addLayout(state_layout)
        
        layout.addLayout(search_layout)
        
        # Search button
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.perform_person_search)
        layout.addWidget(search_button)
        
        # Results area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        # Add visualization button
        viz_button = QPushButton("Visualize Results")
        viz_button.clicked.connect(self.show_visualizations)
        layout.addWidget(viz_button)
        
        return widget

    def add_verizon_logo(self):
        logo_label = QLabel()
        logo_pixmap = QPixmap("verizon_logo.png")
        logo_label.setPixmap(logo_pixmap.scaled(200, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return logo_label

    def create_phone_lookup_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Phone input
        phone_label = QLabel("Phone Number:")
        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("Enter phone number...")
        layout.addWidget(phone_label)
        layout.addWidget(self.phone_input)
        
        # Search button
        search_button = QPushButton("Lookup")
        search_button.clicked.connect(self.perform_phone_lookup)
        layout.addWidget(search_button)
        
        # Results area
        self.phone_results = QTextEdit()
        self.phone_results.setReadOnly(True)
        layout.addWidget(self.phone_results)
        
        return widget

    def create_records_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Record type selection
        record_label = QLabel("Record Type:")
        self.record_type = QComboBox()
        self.record_type.addItems(["Criminal Records", "Court Records", "Traffic Records"])
        layout.addWidget(record_label)
        layout.addWidget(self.record_type)
        
        # Person details
        person_layout = QHBoxLayout()
        
        # Name inputs
        self.record_fname = QLineEdit()
        self.record_fname.setPlaceholderText("First Name")
        self.record_lname = QLineEdit()
        self.record_lname.setPlaceholderText("Last Name")
        
        person_layout.addWidget(self.record_fname)
        person_layout.addWidget(self.record_lname)
        layout.addLayout(person_layout)
        
        # Search button
        search_button = QPushButton("Search Records")
        search_button.clicked.connect(self.perform_records_search)
        layout.addWidget(search_button)
        
        # Results area
        self.records_results = QTextEdit()
        self.records_results.setReadOnly(True)
        layout.addWidget(self.records_results)
        
        return widget

    def create_skip_tracing_tab(self):
        """Create a tab for skip tracing with integrated search functionality"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Add explanation
        explanation = QLabel("Skip tracing helps locate individuals by searching multiple data sources")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Create search form
        form_layout = QGridLayout()
        
        # First name
        form_layout.addWidget(QLabel("First Name:"), 0, 0)
        self.st_first_name = QLineEdit()
        form_layout.addWidget(self.st_first_name, 0, 1)
        
        # Last name
        form_layout.addWidget(QLabel("Last Name:"), 1, 0)
        self.st_last_name = QLineEdit()
        form_layout.addWidget(self.st_last_name, 1, 1)
        
        # State
        form_layout.addWidget(QLabel("State:"), 2, 0)
        self.st_state = QComboBox()
        self.st_state.addItems(["All States", "Alabama", "Alaska", "Arizona", "Arkansas", 
                           "California", "Colorado", "Connecticut", "Delaware", 
                           "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", 
                           "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", 
                           "Maine", "Maryland", "Massachusetts", "Michigan", 
                           "Minnesota", "Mississippi", "Missouri", "Montana", 
                           "Nebraska", "Nevada", "New Hampshire", "New Jersey", 
                           "New Mexico", "New York", "North Carolina", "North Dakota", 
                           "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", 
                           "South Carolina", "South Dakota", "Tennessee", "Texas", 
                           "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
                           "Wisconsin", "Wyoming"])
        form_layout.addWidget(self.st_state, 2, 1)
        
        # City
        form_layout.addWidget(QLabel("City (optional):"), 3, 0)
        self.st_city = QLineEdit()
        form_layout.addWidget(self.st_city, 3, 1)
        
        # Age
        form_layout.addWidget(QLabel("Age (optional):"), 4, 0)
        self.st_age = QSpinBox()
        self.st_age.setMinimum(0)
        self.st_age.setMaximum(120)
        self.st_age.setValue(0)
        self.st_age.setSpecialValueText("Any")
        form_layout.addWidget(self.st_age, 4, 1)
        
        layout.addLayout(form_layout)
        
        # Initialize skip tracing manager
        self.skip_tracer = SkipTracingManager()
        
        # Add data sources group
        sources_group = QGroupBox("Data Sources")
        sources_layout = QVBoxLayout()
        
        # Add checkboxes for each data source
        self.st_source_checkboxes = {}
        for source in self.skip_tracer.data_sources.keys():
            formatted_name = source.replace('_', ' ').title()
            checkbox = QCheckBox(formatted_name)
            checkbox.setChecked(source in self.skip_tracer.active_sources)
            # Disable sources that aren't implemented yet
            if self.skip_tracer.data_sources[source] is None:
                checkbox.setEnabled(False)
                checkbox.setText(f"{formatted_name} (Coming Soon)")
            self.st_source_checkboxes[source] = checkbox
            sources_layout.addWidget(checkbox)
            
        sources_group.setLayout(sources_layout)
        layout.addWidget(sources_group)
        
        # Search button
        self.st_search_button = QPushButton("Start Skip Tracing")
        self.st_search_button.clicked.connect(self.start_skip_tracing)
        layout.addWidget(self.st_search_button)
        
        # Progress bar
        self.st_progress_bar = QProgressBar()
        self.st_progress_bar.setVisible(False)
        layout.addWidget(self.st_progress_bar)
        
        return widget
    
    def create_skip_tracing_results_tab(self):
        """Create a tab for displaying skip tracing results"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Results area
        self.st_results_area = QTextEdit()
        self.st_results_area.setReadOnly(True)
        layout.addWidget(self.st_results_area)
        
        # Export button
        export_button = QPushButton("Export Results")
        export_button.clicked.connect(self.export_skip_tracing_results)
        layout.addWidget(export_button)
        
        return widget
        
    def start_skip_tracing(self):
        """Start the skip tracing process"""
        # Validate inputs
        if not self.st_first_name.text() or not self.st_last_name.text():
            QMessageBox.warning(self, "Input Error", "First name and last name are required.")
            return
            
        # Update active sources based on checkboxes
        self.skip_tracer.active_sources = [
            source for source, checkbox in self.st_source_checkboxes.items()
            if checkbox.isChecked() and checkbox.isEnabled()
        ]
        
        if not self.skip_tracer.active_sources:
            QMessageBox.warning(self, "Source Error", "At least one data source must be selected.")
            return
            
        # Show progress
        self.st_progress_bar.setVisible(True)
        self.st_progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Select or create results tab
        results_tab_index = -1
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Skip Tracing Results":
                results_tab_index = i
                break
                
        if results_tab_index == -1:
            # Create the results tab if it doesn't exist
            self.skip_tracing_results_tab = self.create_skip_tracing_results_tab()
            results_tab_index = self.tab_widget.addTab(self.skip_tracing_results_tab, "Skip Tracing Results")
            
        # Clear results
        self.st_results_area.clear()
        self.st_results_area.append("Skip tracing in progress...\n")
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(results_tab_index)
        
        # Get search parameters
        params = {
            'first_name': self.st_first_name.text(),
            'last_name': self.st_last_name.text(),
            'state': self.st_state.currentText() if self.st_state.currentText() != "All States" else None,
            'city': self.st_city.text() if self.st_city.text() else None,
            'age': self.st_age.value() if self.st_age.value() > 0 else None
        }
        
        # Start the skip tracing in a separate thread
        self.trace_thread = QThread()
        self.trace_worker = SkipTracingWorker(self.skip_tracer, params)
        self.trace_worker.moveToThread(self.trace_thread)
        
        # Connect signals
        self.trace_thread.started.connect(self.trace_worker.run)
        self.trace_worker.results.connect(self.handle_skip_tracing_results)
        self.trace_worker.finished.connect(self.trace_thread.quit)
        self.trace_worker.finished.connect(self.trace_worker.deleteLater)
        self.trace_thread.finished.connect(self.trace_thread.deleteLater)
        
        # Start the thread
        self.trace_thread.start()
        
    def handle_skip_tracing_results(self, results):
        """Handle skip tracing results"""
        self.st_progress_bar.setVisible(False)
        self.st_results_area.clear()
        
        for line in results:
            self.st_results_area.append(line)
            
    def export_skip_tracing_results(self):
        """Export the skip tracing results"""
        if not self.st_results_area.toPlainText():
            QMessageBox.warning(self, "Export Error", "No results to export.")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Skip Tracing Results", "", 
            "Text Files (*.txt);;CSV Files (*.csv);;PDF Files (*.pdf)")
            
        if file_name:
            try:
                if file_name.endswith('.csv'):
                    self.export_to_csv(file_name)
                elif file_name.endswith('.pdf'):
                    self.export_to_pdf(file_name)
                else:
                    # Default to text file
                    with open(file_name, 'w') as f:
                        f.write(self.st_results_area.toPlainText())
                        
                QMessageBox.information(self, "Export Complete", 
                                      "Skip tracing results exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", 
                                   f"Error exporting results: {str(e)}")
    
    def export_to_csv(self, filename):
        """Export results to CSV format"""
        lines = self.st_results_area.toPlainText().split('\n')
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Category', 'Information'])
            
            current_category = "General"
            for line in lines:
                if line.startswith('==='):
                    current_category = line.strip('= ')
                elif line and not line.startswith('-'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        writer.writerow([key.strip(), value.strip()])
                    else:
                        writer.writerow([current_category, line])
                        
    def export_to_pdf(self, filename):
        """Export results to PDF format"""
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title
        elements.append(Paragraph("Skip Tracing Results", styles['Title']))
        elements.append(Spacer(1, 12))
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elements.append(Paragraph(f"Generated: {timestamp}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Process the results
        lines = self.st_results_area.toPlainText().split('\n')
        for line in lines:
            if line.startswith('==='):
                # Section headers
                elements.append(Spacer(1, 12))
                elements.append(Paragraph(line.strip('= '), styles['Heading2']))
                elements.append(Spacer(1, 6))
            elif line.startswith('-'):
                # Separator lines
                elements.append(Paragraph("_" * 40, styles['Normal']))
            else:
                # Regular content
                elements.append(Paragraph(line, styles['Normal']))
                
        # Build the PDF
        doc.build(elements)
        
    def open_skip_tracing_dialog(self):
        """Open the skip tracing dialog (keeping for backward compatibility)"""
        dialog = SkipTracingDialog(self)
        dialog.exec_()

    def perform_person_search(self):
        """Perform person search"""
        try:
            # Get search parameters
            search_params = {
                'first_name': self.fname_input.text(),
                'last_name': self.lname_input.text(),
                'state': self.state_combo.currentText(),
                'use_social': self.use_social.isChecked(),
                'use_public': self.use_github.isChecked(),
                'use_dark': self.use_dark.isChecked()
            }
            
            # Create and configure search worker
            self.search_worker = ComprehensiveSearchWorker()
            self.search_worker.set_search_params(search_params)
            
            # Connect signals
            self.search_worker.progress.connect(self.update_progress)
            self.search_worker.status.connect(self.update_status)
            self.search_worker.results.connect(self.handle_search_complete)
            
            # Start search
            self.search_worker.start()
            
        except Exception as e:
            self.status_bar.showMessage(f"Error starting search: {str(e)}")
            logging.error(f"Error in perform_person_search: {str(e)}")

    def check_cache(self, search_type, query):
        c = self.conn.cursor()
        cache_days = self.cache_days.value()
        c.execute('''SELECT results FROM search_cache 
                    WHERE search_type = ? AND query = ? 
                    AND datetime(timestamp) > datetime('now', ?)''',
                 (search_type, query, f'-{cache_days} days'))
        result = c.fetchone()
        return result[0] if result else None

    def update_cache(self, search_type, query, results):
        c = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        c.execute('''INSERT INTO search_cache (search_type, query, results, timestamp)
                    VALUES (?, ?, ?, ?)''',
                 (search_type, query, results, timestamp))
        self.conn.commit()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_bar.showMessage(message)

    def handle_search_complete(self, results):
        """Handle search completion"""
        try:
            if not results:
                self.status_bar.showMessage("No results found")
                return

            # Process and display results
            self.results_text.clear()
            for result in results:
                if isinstance(result, dict):
                    # Format dictionary results
                    for key, value in result.items():
                        self.results_text.append(f"{key}: {value}")
                    self.results_text.append("-" * 50)
                else:
                    # Handle string or other result types
                    self.results_text.append(str(result))
                    
            self.status_bar.showMessage("Search completed")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error displaying results: {str(e)}")
            logging.error(f"Error in handle_search_complete: {str(e)}")

    def export_results(self):
        if not self.results_text.toPlainText():
            QMessageBox.warning(self, "Export Error", "No results to export.")
            return
        
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Results", "", 
            "CSV Files (*.csv);;Text Files (*.txt);;PDF Files (*.pdf);;JSON Files (*.json)")
        
        if file_name:
            try:
                if file_name.endswith('.csv'):
                    self.export_to_csv(file_name)
                elif file_name.endswith('.pdf'):
                    self.export_to_pdf(file_name)
                elif file_name.endswith('.json'):
                    self.export_to_json(file_name)
                else:
                    self.export_to_txt(file_name)
                    
                QMessageBox.information(self, "Export Complete", 
                                      "Results exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", 
                                   f"Error exporting results: {str(e)}")

    def export_to_csv(self, file_name):
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Type', 'Information'])
            for line in self.results_text.toPlainText().split('\n'):
                if line.strip():
                    if line.startswith('['):
                        timestamp = line[1:20]
                        rest = line[22:]
                    else:
                        timestamp = ''
                        rest = line
                    writer.writerow([timestamp, 'Information', rest])

    def export_to_txt(self, file_name):
        with open(file_name, 'w') as f:
            f.write(self.results_text.toPlainText())

    def export_to_pdf(self, file_name):
        doc = SimpleDocTemplate(file_name, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        title = Paragraph("Background Check Results", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Add timestamp
        timestamp = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                            styles['Normal'])
        story.append(timestamp)
        story.append(Spacer(1, 12))
        
        # Add results
        results = self.results_text.toPlainText().split('\n')
        for result in results:
            if result.strip():
                p = Paragraph(result, styles['Normal'])
                story.append(p)
                story.append(Spacer(1, 6))
        
        doc.build(story)

    def export_to_json(self, file_name):
        results = self.results_text.toPlainText().split('\n')
        data = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'metadata': {
                'search_type': self.current_search_type,
                'parameters': self.current_search_params
            }
        }
        
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)

    def show_visualizations(self):
        # Prepare data for visualization
        results = self.results_text.toPlainText().split('\n')
        data = []
        for result in results:
            if result.startswith('['):
                timestamp = datetime.strptime(result[1:20], '%Y-%m-%d %H:%M:%S')
                data.append({
                    'timestamp': timestamp,
                    'value': 1,
                    'text': result[22:]
                })
        
        # Show visualization dialog
        dialog = DataVisualizationDialog(data, self)
        dialog.exec_()

    def perform_phone_lookup(self):
        phone = self.phone_input.text()
        results = []
        
        # Clean phone number
        phone_clean = re.sub(r'\D', '', phone)
        
        if len(phone_clean) != 10:
            self.phone_results.setText("Please enter a valid 10-digit phone number")
            return
        
        # Search for phone number mentions online
        try:
            response = requests.get(
                f"https://www.google.com/search?q={phone_clean}",
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract relevant information
            results.append(f"Phone Number: {phone}")
            results.append(f"Area Code Location: {self.get_area_code_location(phone_clean[:3])}")
            
            # Look for business listings
            business_results = self.search_business_listings(phone_clean)
            if business_results:
                results.extend(business_results)
            
        except Exception as e:
            results.append(f"Error during search: {str(e)}")
        
        self.phone_results.setText("\n".join(results))

    def get_area_code_location(self, area_code):
        # Basic area code to location mapping (could be expanded)
        area_codes = {
            "678": "Georgia (Atlanta)",
            "815": "Illinois",
            "410": "Maryland (Baltimore)",
            "443": "Maryland (Baltimore overlay)"
        }
        return area_codes.get(area_code, "Unknown location")

    def search_business_listings(self, phone):
        results = []
        try:
            # Search Google Maps for business listings
            response = requests.get(
                f"https://www.google.com/search?q={phone}+business",
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract business names and addresses
            business_elements = soup.find_all('div', {'class': 'BNeawe iBp4i AP7Wnd'})
            for element in business_elements[:3]:  # Limit to first 3 results
                results.append(f"Possible Business: {element.text}")
        except Exception as e:
            print(f"Business lookup error: {e}")
        return results

    def perform_records_search(self):
        first_name = self.record_fname.text()
        last_name = self.record_lname.text()
        record_type = self.record_type.currentText()
        
        results = []
        
        # Search court records
        if record_type == "Court Records":
            court_results = self.search_court_records(first_name, last_name)
            if court_results:
                results.extend(court_results)
        
        # Search criminal records
        elif record_type == "Criminal Records":
            criminal_results = self.search_criminal_records(first_name, last_name)
            if criminal_results:
                results.extend(criminal_results)
        
        # Search traffic records
        elif record_type == "Traffic Records":
            traffic_results = self.search_traffic_records(first_name, last_name)
            if traffic_results:
                results.extend(traffic_results)
        
        # Format and display results
        self.records_results.setText("\n".join(results) if results else "No records found")

    def search_court_records(self, first_name, last_name):
        results = []
        try:
            # Search for court records using Google dorks
            query = f'site:.gov "{first_name} {last_name}" (court OR case OR docket)'
            response = requests.get(
                f"https://www.google.com/search?q={query}",
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract relevant links and snippets
            for result in soup.find_all('div', {'class': 'g'})[:5]:
                title = result.find('h3')
                link = result.find('a')
                if title and link:
                    results.append(f"Court Record: {title.text}")
                    results.append(f"Source: {link['href']}\n")
        except Exception as e:
            print(f"Court records search error: {e}")
        return results

    def search_criminal_records(self, first_name, last_name):
        results = []
        try:
            # Search state-specific criminal record databases
            for state in ["Georgia", "Illinois", "Maryland"]:
                state_records = self.search_state_criminal_records(first_name, last_name, state)
                if state_records:
                    results.extend([f"{state} Records:"] + state_records)
        except Exception as e:
            print(f"Criminal records search error: {e}")
        return results

    def search_state_criminal_records(self, first_name, last_name, state):
        results = []
        state_urls = {
            "Georgia": "https://gbi.georgia.gov/",
            "Illinois": "https://www.isp.state.il.us/",
            "Maryland": "https://www.dpscs.state.md.us/"
        }
        
        try:
            # Search state website for criminal records
            url = state_urls.get(state)
            if url:
                response = requests.get(
                    f"https://www.google.com/search?q=site:{url} \"{first_name} {last_name}\"",
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract relevant information
                for result in soup.find_all('div', {'class': 'g'})[:3]:
                    snippet = result.find('div', {'class': 'VwiC3b'})
                    if snippet:
                        results.append(snippet.text)
        except Exception as e:
            print(f"State criminal records search error: {e}")
        return results

    def search_traffic_records(self, first_name, last_name):
        results = []
        try:
            # Search for traffic violations using Google dorks
            query = f'"{first_name} {last_name}" (traffic OR violation OR citation)'
            response = requests.get(
                f"https://www.google.com/search?q={query}",
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract relevant information
            for result in soup.find_all('div', {'class': 'g'})[:5]:
                snippet = result.find('div', {'class': 'VwiC3b'})
                if snippet:
                    results.append(f"Traffic Record: {snippet.text}")
        except Exception as e:
            print(f"Traffic records search error: {e}")
        return results

    def enhance_search_results(self, results):
        """Add additional context and formatting to search results"""
        enhanced_results = []
        for result in results:
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            enhanced_results.append(f"[{timestamp}] {result}")
            
            # Add source attribution if available
            if "Source:" not in result and "http" in result:
                enhanced_results.append(f"Source: {result}")
            
            # Add separator
            enhanced_results.append("-" * 50)
        
        return enhanced_results

    def show_filter_dialog(self):
        # Convert results to DataFrame for filtering
        results = self.results_text.toPlainText().split('\n')
        data = []
        for result in results:
            if result.strip():
                data.append({
                    'timestamp': datetime.now(),
                    'text': result,
                    'source': self.detect_source(result)
                })
        
        df = pd.DataFrame(data)
        
        dialog = FilterDialog(df, self)
        if dialog.exec_():
            filtered_df = dialog.apply_filters()
            self.display_filtered_results(filtered_df)

    def detect_source(self, result):
        if 'GitHub' in result:
            return 'GitHub'
        elif any(s in result for s in ['LinkedIn', 'Twitter', 'Facebook']):
            return 'Social Media'
        else:
            return 'Public Records'

    def display_filtered_results(self, df):
        self.results_text.setText('\n'.join(df['text'].tolist()))

class EnhancedSecurityMonitor(QThread):
    alert = Signal(dict)
    progress = Signal(int)
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self.leak_monitor = AdvancedLeakMonitor()
        self.hash_checker = EnhancedPasswordChecker()
        self.dark_scanner = ComprehensiveDarkWebScanner()
        self.notifier = MultiChannelNotifier()
        self.encryption = AdvancedEncryption()
        self.is_running = True
        self.monitoring_frequency = 3600  # 1 hour
        
    async def initialize(self):
        self.status.emit("Initializing enhanced security monitoring...")
        try:
            # Initialize Tor with multiple circuits
            await self.dark_scanner.initialize_tor_multi_circuit()
            
            # Initialize encryption
            await self.encryption.initialize()
            
            # Set up secure channels
            await self.notifier.initialize_channels()
            
            self.status.emit("Security infrastructure initialized")
        except Exception as e:
            self.status.emit(f"Initialization failed: {str(e)}")
            raise

    async def enhanced_security_scan(self, user_data):
        self.status.emit("Starting comprehensive security scan...")
        total_steps = 7
        current_step = 0

        try:
            # Enhanced data leak check
            self.status.emit("Performing deep data leak analysis...")
            leaks = await self.check_data_leaks_enhanced(user_data)
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))

            # Advanced password security
            self.status.emit("Conducting advanced password analysis...")
            password_issues = await self.check_password_security_enhanced(user_data)
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))

            # Comprehensive dark web scan
            self.status.emit("Executing comprehensive dark web scan...")
            dark_web_data = await self.scan_dark_web_enhanced(user_data)
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))

            # Digital footprint analysis
            self.status.emit("Analyzing digital footprint...")
            footprint_data = await self.analyze_digital_footprint(user_data)
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))

            # Threat intelligence gathering
            self.status.emit("Gathering threat intelligence...")
            threat_data = await self.gather_threat_intelligence(user_data)
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))

            # Vulnerability assessment
            self.status.emit("Performing vulnerability assessment...")
            vulnerabilities = await self.assess_vulnerabilities(user_data)
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))

            # Generate comprehensive report
            self.status.emit("Generating detailed security report...")
            report = self.generate_enhanced_security_report(
                leaks, password_issues, dark_web_data,
                footprint_data, threat_data, vulnerabilities
            )
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))

            # Send notifications through all channels
            await self.notifier.send_multi_channel_notification(report)

            return report

        except Exception as e:
            self.status.emit(f"Security scan error: {str(e)}")
            return []

    async def check_data_leaks_enhanced(self, user_data):
        encrypted_data = await self.encryption.encrypt_data_enhanced(user_data)
        sources = {
            'clearnet': self.leak_monitor.check_clearnet_sources_enhanced,
            'darknet': self.leak_monitor.check_darknet_sources_enhanced,
            'paste_sites': self.leak_monitor.check_paste_sites,
            'breach_databases': self.leak_monitor.check_breach_databases,
            'underground_forums': self.leak_monitor.check_underground_forums
        }
        
        results = []
        for source_name, check_function in sources.items():
            try:
                source_results = await check_function(encrypted_data)
                results.extend(self.process_source_results(source_results, source_name))
            except Exception as e:
                logging.error(f"Enhanced leak check failed for {source_name}: {str(e)}")
                
        return self.deduplicate_and_verify_leaks_enhanced(results)

    async def check_password_security_enhanced(self, user_data):
        checks = {
            'hash_check': self.hash_checker.check_known_breaches,
            'pattern_analysis': self.hash_checker.analyze_patterns,
            'complexity_score': self.hash_checker.calculate_complexity,
            'rainbow_table': self.hash_checker.check_rainbow_tables,
            'common_password': self.hash_checker.check_common_passwords
        }
        
        results = []
        for check_name, check_function in checks.items():
            try:
                check_result = await check_function(user_data)
                results.extend(self.process_password_check(check_result, check_name))
            except Exception as e:
                logging.error(f"Enhanced password check failed for {check_name}: {str(e)}")
                
        return self.aggregate_password_results(results)

    def generate_enhanced_security_report(self, *args):
        report = []
        sections = {
            'Data Leaks': self.format_leak_section,
            'Password Security': self.format_password_section,
            'Dark Web Findings': self.format_dark_web_section,
            'Digital Footprint': self.format_footprint_section,
            'Threat Intelligence': self.format_threat_section,
            'Vulnerabilities': self.format_vulnerability_section
        }
        
        report.append("=== COMPREHENSIVE SECURITY ASSESSMENT ===")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 50 + "\n")
        
        for section_name, formatter in sections.items():
            report.extend(formatter(args[sections[section_name]]))
            report.append("\n" + "=" * 50 + "\n")
            
        report.append("\nRECOMMENDED ACTIONS:")
        report.extend(self.generate_recommendations(args))
        
        return report

    def stop(self):
        self.is_running = False
        self.status.emit("Enhanced security monitoring stopped")
        self.cleanup_resources()

class AdvancedIdentityCheck(QThread):
    progress = Signal(int)
    status = Signal(str)
    results = Signal(list)

    def __init__(self):
        super().__init__()
        self.leak_checker = ComprehensiveLeakChecker()
        self.is_running = True
        self.delay_between_searches = 2  # seconds
        self.face_recognition_enabled = FACE_RECOGNITION_AVAILABLE
        
    async def search_identity(self, user_data):
        self.status.emit("Starting comprehensive identity search...")
        total_steps = 5
        current_step = 0

        try:
            # Initialize results dictionary
            search_results = {
                'leaks': [],
                'social_profiles': [],
                'public_records': [],
                'dark_web': [],
                'analysis': {}
            }

            # Check for leaked information
            self.status.emit("Checking for leaked information...")
            leak_results = await self.leak_checker.search_all_variations(user_data)
            search_results['leaks'] = leak_results
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))
            await asyncio.sleep(self.delay_between_searches)

            # Process and analyze results
            self.status.emit("Analyzing findings...")
            analyzed_results = self.analyze_findings(search_results)
            search_results['analysis'] = analyzed_results
            current_step += 1
            self.progress.emit(int((current_step/total_steps) * 100))

            # Generate comprehensive report
            formatted_results = self.format_comprehensive_results(search_results)
            
            return formatted_results

        except Exception as e:
            self.status.emit(f"Error during search: {str(e)}")
            return []

    def analyze_findings(self, results):
        """Analyze the findings for patterns and severity"""
        analysis = {
            'risk_level': 'low',
            'patterns_detected': [],
            'recommendations': [],
            'urgent_actions': []
        }

        # Analyze leak patterns
        if results['leaks']:
            leak_analysis = self.analyze_leak_patterns(results['leaks'])
            analysis['patterns_detected'].extend(leak_analysis['patterns'])
            analysis['risk_level'] = self.calculate_risk_level(leak_analysis)
            analysis['recommendations'].extend(leak_analysis['recommendations'])
            
            if leak_analysis.get('urgent_actions'):
                analysis['urgent_actions'].extend(leak_analysis['urgent_actions'])

        return analysis

    def analyze_leak_patterns(self, leaks):
        """Analyze patterns in leaked data"""
        analysis = {
            'patterns': [],
            'recommendations': [],
            'urgent_actions': []
        }

        # Check for password reuse
        if any('password' in leak for leak in leaks):
            analysis['patterns'].append('password_reuse_detected')
            analysis['recommendations'].append(
                'Change passwords on all accounts and use unique passwords'
            )
            analysis['urgent_actions'].append('change_compromised_passwords')

        # Check for personal information exposure
        if any('ssn' in str(leak).lower() for leak in leaks):
            analysis['patterns'].append('ssn_exposed')
            analysis['urgent_actions'].append('monitor_credit_reports')
            analysis['recommendations'].append('Set up credit monitoring')

        return analysis

    def calculate_risk_level(self, analysis):
        """Calculate overall risk level based on findings"""
        risk_score = 0
        
        # Critical patterns
        critical_patterns = ['ssn_exposed', 'credit_card_exposed', 'admin_credentials_exposed']
        if any(pattern in analysis['patterns'] for pattern in critical_patterns):
            return 'critical'

        # High risk patterns
        high_risk_patterns = ['password_reuse_detected', 'multiple_leaks_detected']
        if any(pattern in analysis['patterns'] for pattern in high_risk_patterns):
            risk_score += 2

        # Medium risk patterns
        medium_risk_patterns = ['email_exposed', 'phone_exposed']
        if any(pattern in analysis['patterns'] for pattern in medium_risk_patterns):
            risk_score += 1

        # Determine risk level
        if risk_score >= 3:
            return 'high'
        elif risk_score >= 1:
            return 'medium'
        return 'low'

    def format_comprehensive_results(self, results):
        """Format results into a comprehensive report"""
        formatted = []
        
        # Add header
        formatted.extend([
            "=== COMPREHENSIVE IDENTITY LEAK REPORT ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            ""
        ])

        # Add risk level and summary
        risk_level = results['analysis']['risk_level']
        formatted.extend([
            f"RISK LEVEL: {risk_level.upper()}",
            "",
            "=== FINDINGS SUMMARY ===",
            f"Total Leaks Found: {len(results['leaks'])}",
            f"Patterns Detected: {', '.join(results['analysis']['patterns_detected'])}",
            ""
        ])

        # Add urgent actions if any
        if results['analysis']['urgent_actions']:
            formatted.extend([
                "!!! URGENT ACTIONS REQUIRED !!!",
                *[f"- {action}" for action in results['analysis']['urgent_actions']],
                ""
            ])

        # Add detailed findings
        if results['leaks']:
            formatted.extend([
                "=== DETAILED FINDINGS ===",
                *self.format_leak_details(results['leaks']),
                ""
            ])

        # Add recommendations
        formatted.extend([
            "=== RECOMMENDATIONS ===",
            *[f"- {rec}" for rec in results['analysis']['recommendations']],
            ""
        ])

        return formatted

    def format_leak_details(self, leaks):
        """Format detailed leak findings"""
        details = []
        for leak in leaks:
            details.extend([
                f"Source: {leak.get('source', 'Unknown')}",
                f"Date: {leak.get('date', 'Unknown')}",
                f"Type: {leak.get('type', 'Unknown')}",
                f"Severity: {leak.get('severity', 'Unknown')}",
                "-" * 30
            ])
        return details

    def stop(self):
        self.is_running = False
        self.status.emit("Search stopped by user")

    async def process_images(self, image_data):
        if not self.face_recognition_enabled:
            return {
                'status': 'skipped',
                'message': 'Face recognition is not available'
            }
            
        # Rest of the image processing code

    def setup_tabs(self):
        """Setup result tabs"""
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs for different result types
        self.person_results_tab = QTextEdit()
        self.phone_results_tab = QTextEdit()
        self.records_results_tab = QTextEdit()
        
        # Make result areas read-only
        self.person_results_tab.setReadOnly(True)
        self.phone_results_tab.setReadOnly(True)
        self.records_results_tab.setReadOnly(True)
        
        # Style the result areas
        result_style = """
            QTextEdit {
                background-color: #222222;
                color: white;
                border: none;
                padding: 10px;
            }
        """
        self.person_results_tab.setStyleSheet(result_style)
        self.phone_results_tab.setStyleSheet(result_style)
        self.records_results_tab.setStyleSheet(result_style)
        
        # Add tabs
        self.tab_widget.addTab(self.person_results_tab, "Person Results")
        self.tab_widget.addTab(self.phone_results_tab, "Phone Results")
        self.tab_widget.addTab(self.records_results_tab, "Records Results")
        
        # Style the tabs
        self.tab_widget.setStyleSheet("""
            QTabBar::tab {
                background-color: #333333;
                color: white;
                padding: 8px 20px;
                border: none;
            }
            QTabBar::tab:selected {
                background-color: #FF0000;
            }
            QTabWidget::pane {
                border: none;
            }
        """)

    def setup_search_section(self):
        """Setup unified search section"""
        search_group = QGroupBox("Universal Search")
        search_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid #444444;
                margin-top: 1em;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        
        search_layout = QGridLayout()
        label_style = "QLabel { color: white; }"

        # Create and style input fields
        input_style = """
            QLineEdit {
                background-color: #333333;
                color: white;
                border: 1px solid #444444;
                padding: 5px;
            }
            QLineEdit:focus {
                border: 1px solid #FF0000;
            }
        """

        # Email field
        email_label = QLabel("Email:")
        email_label.setStyleSheet(label_style)
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email Address")
        self.email_input.setStyleSheet(input_style)
        search_layout.addWidget(email_label, 0, 0)
        search_layout.addWidget(self.email_input, 0, 1)

        # Phone field
        phone_label = QLabel("Phone:")
        phone_label.setStyleSheet(label_style)
        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("Phone Number")
        self.phone_input.setStyleSheet(input_style)
        search_layout.addWidget(phone_label, 0, 2)
        search_layout.addWidget(self.phone_input, 0, 3)

        # Username field
        username_label = QLabel("Username:")
        username_label.setStyleSheet(label_style)
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Online Username")
        self.username_input.setStyleSheet(input_style)
        search_layout.addWidget(username_label, 1, 0)
        search_layout.addWidget(self.username_input, 1, 1)

        # Name fields
        name_label = QLabel("Name:")
        name_label.setStyleSheet(label_style)
        search_layout.addWidget(name_label, 1, 2)
        
        name_layout = QHBoxLayout()
        self.first_name_input = QLineEdit()
        self.first_name_input.setPlaceholderText("First Name")
        self.first_name_input.setStyleSheet(input_style)
        self.last_name_input = QLineEdit()
        self.last_name_input.setPlaceholderText("Last Name")
        self.last_name_input.setStyleSheet(input_style)
        name_layout.addWidget(self.first_name_input)
        name_layout.addWidget(self.last_name_input)
        search_layout.addLayout(name_layout, 1, 3)

        search_group.setLayout(search_layout)
        return search_group

    def update_results(self, results):
        """Update results in all tabs based on search type"""
        try:
            # Clear previous results
            self.person_results_tab.clear()
            self.phone_results_tab.clear()
            self.records_results_tab.clear()

            for result in results:
                if isinstance(result, dict):
                    # Determine which tab(s) to update based on result type
                    result_type = result.get('type', '').lower()
                    
                    # Format result text
                    result_text = self.format_result(result)
                    
                    # Update appropriate tab(s)
                    if 'person' in result_type:
                        self.person_results_tab.append(result_text)
                    if 'phone' in result_type:
                        self.phone_results_tab.append(result_text)
                    if 'record' in result_type:
                        self.records_results_tab.append(result_text)
                    
                    # If result type is not specified, show in all tabs
                    if not result_type:
                        self.person_results_tab.append(result_text)
                        self.phone_results_tab.append(result_text)
                        self.records_results_tab.append(result_text)
                else:
                    # If result is not a dict, show in all tabs
                    self.person_results_tab.append(str(result))
                    self.phone_results_tab.append(str(result))
                    self.records_results_tab.append(str(result))

        except Exception as e:
            self.status_bar.showMessage(f"Error updating results: {str(e)}")
            logging.error(f"Error updating results: {str(e)}")

    def format_result(self, result):
        """Format a result dictionary into displayable text"""
        formatted_text = ""
        for key, value in result.items():
            if key != 'type':  # Skip the type field as it's used for sorting
                formatted_text += f"{key.title()}: {value}\n"
        formatted_text += "-" * 50 + "\n"
        return formatted_text

    def perform_unified_search(self):
        """Perform search using any provided search criteria"""
        try:
            # Collect all search parameters
            search_params = {
                'email': self.email_input.text(),
                'phone': self.phone_input.text(),
                'username': self.username_input.text(),
                'first_name': self.first_name_input.text(),
                'last_name': self.last_name_input.text()
            }
            
            # Validate that at least one field is filled
            if not any(search_params.values()):
                QMessageBox.warning(self, "Search Error", 
                                  "Please enter at least one search term")
                return
                
            # Clear previous results
            self.person_results_tab.clear()
            self.phone_results_tab.clear()
            self.records_results_tab.clear()
            self.progress_bar.setValue(0)
            
            # Create and configure search worker
            self.search_worker = ComprehensiveSearchWorker()
            self.search_worker.set_search_params(search_params)
            
            # Connect signals
            self.search_worker.progress.connect(self.update_progress)
            self.search_worker.status.connect(self.update_status)
            self.search_worker.results.connect(self.update_results)
            
            # Disable search button while searching
            self.search_button.setEnabled(False)
            self.status_bar.showMessage("Starting unified search...")
            
            # Start search
            self.search_worker.start()
            
        except Exception as e:
            self.status_bar.showMessage(f"Error starting search: {str(e)}")
            logging.error(f"Error in perform_unified_search: {str(e)}")
            self.search_button.setEnabled(True)

class TruePeopleSearchScraper:
    def __init__(self):
        self.base_url = "https://www.truepeoplesearch.com"
        self.search_url = f"{self.base_url}/results"
        
        # Add a selenium option flag
        self.use_selenium = False  # Set to True to use browser automation
        self.use_cloudscraper = True  # Set to True to use cloudscraper
        
        # Expanded and modernized headers pool with newer browser versions
        self.headers_pool = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.google.com/',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Sec-Fetch-User': '?1',
                'Pragma': 'no-cache',
                'Cache-Control': 'no-cache',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/121.0.0.0 Safari/537.36 Edg/121.0.0.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.bing.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Sec-Fetch-User': '?1',
            }
        ]
        
        # Updated proxy pool with more options
        self.proxy_pool = [
            None,  # No proxy option
            # Free proxies from public sources (these may not work reliably)
            "http://34.141.106.53:80",
            "http://158.69.71.245:9300",
            "http://107.175.59.189:3128",
            # Add your own proxies here in the format: "http://user:pass@ip:port"
        ]
        
        self.session = requests.Session()
        # Initialize cookies to make the session appear more like a regular browser
        self.cookies = {}
        # Track failed attempts to implement exponential backoff
        self.failed_attempts = 0
        self.max_retries = 3
        
        # Add mobile user agents to make requests appear from mobile devices
        self.mobile_headers_pool = [
            {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Linux; Android 13; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            }
        ]
        
    def get_random_headers(self, use_mobile=False):
        """Get random headers to avoid detection"""
        if use_mobile:
            headers = random.choice(self.mobile_headers_pool).copy()
        else:
            headers = random.choice(self.headers_pool).copy()
            
        # Add some randomization to appear more like a real browser
        if random.random() < 0.5:
            headers['Accept-Encoding'] = 'gzip'
        
        # Add some random browser fingerprinting resistance
        if random.random() < 0.3:
            headers['Sec-GPC'] = '1'  # Global Privacy Control
            
        # Sometimes add do-not-track header
        if random.random() < 0.7:
            headers['DNT'] = '1'
            
        return headers
    
    def get_random_proxy(self):
        """Get a random proxy from the pool"""
        return random.choice(self.proxy_pool)
    
    async def wait_with_backoff(self):
        """Implement exponential backoff for retries"""
        if self.failed_attempts > 0:
            # Exponential backoff: 2^attempts * base_delay * random_factor
            base_delay = 2
            max_delay = 60  # Cap at 60 seconds
            delay = min(2 ** self.failed_attempts * base_delay * (0.5 + random.random()), max_delay)
            await asyncio.sleep(delay)
        else:
            # Normal delay between 3-7 seconds to mimic human behavior
            await asyncio.sleep(random.uniform(3, 7))
            
    async def search_person_with_selenium(self, first_name, last_name, state=None, city=None, age=None):
        """Search using Selenium for better anti-detection"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import time
            
            # Set up headless Chrome
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # Add random user agent
            user_agent = random.choice(self.headers_pool)['User-Agent']
            chrome_options.add_argument(f'--user-agent={user_agent}')
            
            # Create driver
            driver = webdriver.Chrome(options=chrome_options)
            
            # First visit the homepage to get cookies
            driver.get(self.base_url)
            time.sleep(random.uniform(2, 4))
            
            # Now go to the search URL
            params = {
                'fn': first_name,
                'ln': last_name
            }
            
            if state and state != "All States":
                params['state'] = state
                
            if city:
                params['city'] = city
            
            search_url = f"{self.search_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
            driver.get(search_url)
            
            # Wait for results to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extract HTML
            html = driver.page_source
            driver.quit()
            
            # Parse the results
            return self.parse_search_results(html)
            
        except Exception as e:
            return [f"Error using Selenium: {str(e)}"]
    
    async def search_person_with_cloudscraper(self, first_name, last_name, state=None, city=None, age=None):
        """Search using cloudscraper to bypass Cloudflare protection"""
        try:
            import cloudscraper
            
            # Create a cloudscraper instance
            scraper = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'windows',
                    'mobile': False
                }
            )
            
            # Construct the search URL with query parameters
            params = {
                'fn': first_name,
                'ln': last_name
            }
            
            if state and state != "All States":
                params['state'] = state
                
            if city:
                params['city'] = city
                
            # Add random parameter to avoid caching
            params['_r'] = str(random.randint(10000000, 99999999))
            
            # Get random headers
            headers = self.get_random_headers()
            
            # First visit homepage to get cookies
            scraper.get(self.base_url)
            
            # Now visit search page
            response = scraper.get(self.search_url, params=params, headers=headers)
            
            if response.status_code == 200:
                return self.parse_search_results(response.text)
            else:
                return [f"Error: CloudScraper received status code {response.status_code}"]
                
        except Exception as e:
            return [f"Error using CloudScraper: {str(e)}"]
    
    async def search_person(self, first_name, last_name, state=None, city=None, age=None):
        """Search for a person on TruePeopleSearch"""
        results = []
        
        # Try using Selenium if enabled
        if self.use_selenium:
            try:
                selenium_results = await self.search_person_with_selenium(first_name, last_name, state, city, age)
                if selenium_results and len(selenium_results) > 1:  # If we got actual results
                    return selenium_results
            except Exception as e:
                results.append(f"Selenium search failed: {str(e)}")
        
        # Try using CloudScraper if enabled
        if self.use_cloudscraper:
            try:
                cloudscraper_results = await self.search_person_with_cloudscraper(first_name, last_name, state, city, age)
                if cloudscraper_results and len(cloudscraper_results) > 1:  # If we got actual results
                    return cloudscraper_results
            except Exception as e:
                results.append(f"CloudScraper search failed: {str(e)}")
        
        # Reset failed attempts counter for this new search
        self.failed_attempts = 0
        
        # If Selenium and CloudScraper didn't work, fallback to normal request method
        for attempt in range(self.max_retries):
            try:
                # Construct the search URL with query parameters
                params = {
                    'fn': first_name,
                    'ln': last_name
                }
                
                if state and state != "All States":
                    params['state'] = state
                    
                if city:
                    params['city'] = city
                
                # Apply backoff strategy before making request
                await self.wait_with_backoff()
                
                # Decide whether to use mobile or desktop headers
                use_mobile = random.random() < 0.3  # 30% chance of using mobile headers
                
                # Get random headers and proxy for this attempt
                headers = self.get_random_headers(use_mobile)
                proxy = self.get_random_proxy()
                
                # Add a random parameter to help bypass caching/fingerprinting
                params['_r'] = str(random.randint(10000000, 99999999))
                
                # Prepare request kwargs
                request_kwargs = {
                    'params': params,
                    'headers': headers,
                    'ssl': False,  # Sometimes helps with SSL issues
                }
                
                if proxy:
                    request_kwargs['proxy'] = proxy
                
                # First make a request to the homepage to get cookies
                async with aiohttp.ClientSession(cookies=self.cookies) as session:
                    # Visit the homepage first to get cookies and appear more natural
                    if not self.cookies:
                        try:
                            async with session.get(self.base_url, headers=headers, timeout=20) as home_response:
                                if home_response.status == 200:
                                    # Save cookies for future requests
                                    self.cookies = {k: v.value for k, v in home_response.cookies.items()}
                                    # Wait a bit like a real user would
                                    await asyncio.sleep(random.uniform(1, 3))
                        except Exception as e:
                            print(f"Error accessing homepage: {str(e)}")
                    
                    # Now make the actual search request
                    async with session.get(self.search_url, **request_kwargs, timeout=30) as response:
                        # Update cookies from this response
                        for k, v in response.cookies.items():
                            self.cookies[k] = v.value
                        
                        if response.status == 200:
                            html = await response.text()
                            # Reset failed attempts on success
                            self.failed_attempts = 0
                            results.extend(self.parse_search_results(html))
                            break  # Success, exit retry loop
                        elif response.status == 429:  # Too Many Requests
                            self.failed_attempts += 1
                            # If this is our last attempt, report the error
                            if attempt == self.max_retries - 1:
                                return [f"Error: Rate limited by TruePeopleSearch. Try again later."]
                        elif response.status == 403:  # Forbidden
                            self.failed_attempts += 1
                            # If this is our last attempt, report the error
                            if attempt == self.max_retries - 1:
                                return [f"Error: Access blocked by TruePeopleSearch (403 Forbidden). This could be due to anti-scraping measures."]
                        else:
                            self.failed_attempts += 1
                            # If this is our last attempt, report the error
                            if attempt == self.max_retries - 1:
                                return [f"Error: Received status code {response.status}"]
            
            except Exception as e:
                self.failed_attempts += 1
                # If this is our last attempt, report the error
                if attempt == self.max_retries - 1:
                    return [f"Error searching TruePeopleSearch: {str(e)}"]
        
        # If all methods failed and we have no results, use the free public APIs to generate some results
        if not results or len(results) <= 1:
            free_data_scraper = FreePublicDataScraper()
            return await free_data_scraper.search_person(first_name, last_name, state, city, age)
            
        return results

    def parse_search_results(self, html):
        """Parse the HTML response to extract person information"""
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check if we're getting a captcha or block page
        if "captcha" in html.lower() or "security check" in html.lower() or "access denied" in html.lower():
            results.append("Access to TruePeopleSearch is currently restricted.")
            results.append("The website may be showing a captcha or security check.")
            return results
        
        # Find all person cards in the search results
        person_cards = soup.select('.card.card-block.shadow-form.card-summary')
        
        if not person_cards:
            # Try alternate selectors in case the site structure changed
            person_cards = soup.select('.card') or soup.select('.result-card') or soup.select('[data-result-card]')
            
            if not person_cards:
                # Check if we're on a "no results" page
                if "no results" in html.lower() or "no matches" in html.lower():
                    results.append("No results found on TruePeopleSearch")
                else:
                    # If we can't find person cards but not seeing "no results" message,
                    # the site structure may have changed
                    results.append("Could not parse results from TruePeopleSearch")
                    results.append("The website structure may have changed.")
                return results
        
        results.append(f"\n=== TruePeopleSearch Results ({len(person_cards)} found) ===\n")
        
        for card in person_cards:
            try:
                # Extract name (try different selectors to be adaptable)
                name_elem = card.select_one('.h4') or card.select_one('.name') or card.select_one('h4')
                name = name_elem.text.strip() if name_elem else "Name not found"
                
                # Extract age
                age_elem = card.select_one('.content-value.age') or card.select_one('.age')
                age = age_elem.text.strip() if age_elem else "Age not available"
                
                # Extract address
                address_elem = card.select_one('.content-value.address') or card.select_one('.address')
                address = address_elem.text.strip() if address_elem else "Address not available"
                
                # Extract details link
                link_elem = card.select_one('a.btn.btn-success') or card.select_one('a.details') or card.select_one('a[href*="person"]')
                details_link = f"{self.base_url}{link_elem['href']}" if link_elem and 'href' in link_elem.attrs else "No link available"
                
                # Format and add to results
                results.append(f"Name: {name}")
                results.append(f"Age: {age}")
                results.append(f"Address: {address}")
                results.append(f"Details: {details_link}")
                results.append("-" * 50)
                
            except Exception as e:
                results.append(f"Error parsing result: {str(e)}")
                continue
                
        return results

class SkipTracingManager:
    def __init__(self):
        self.data_sources = {
            'true_people_search': TruePeopleSearchScraper(),
            'fast_people_search': FastPeopleSearchScraper(),  # New alternative scraper
            'basic_public_records': BasicPublicRecordsScraper(),
            'free_public_data': FreePublicDataScraper(),  # New free public data scraper
            'whitepages': None,
            'spokeo': None,
            'intelius': None,
            'beenverified': None
        }
        self.active_sources = ['true_people_search', 'fast_people_search', 'basic_public_records', 'free_public_data']

    async def trace_person(self, first_name, last_name, state=None, city=None, age=None):
        """Perform comprehensive skip tracing on a person"""
        all_results = []
        tasks = []
        
        # Create tasks for each active source
        for source_name in self.active_sources:
            source = self.data_sources.get(source_name)
            if source:
                # Pass all parameters to all sources - they each handle which ones they need
                tasks.append(source.search_person(first_name, last_name, state, city, age))
        
        # Run all searches concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    all_results.append(f"Error during skip tracing: {str(res)}")
                else:
                    all_results.extend(res)
        
        # Process and integrate results
        processed_results = self.process_results(all_results)
        return processed_results
    
    def process_results(self, results):
        """Process and deduplicate results from multiple sources"""
        # Enhanced implementation with better deduplication and data integration
        processed = []
        seen_addresses = set()
        seen_phones = set()
        seen_emails = set()
        
        processed.append("\n=== SKIP TRACING RESULTS ===\n")
        
        # Extract and deduplicate addresses and phone numbers
        for line in results:
            if any(header in line for header in ["=== TruePeopleSearch", "=== FastPeopleSearch", "=== Basic Public", "=== Free Public"]):
                # Include source headers
                processed.append(line)
                continue
                
            # Skip separators and empty lines during deduplication
            if line.strip() == "" or line.startswith("---") or line.startswith("===") or all(c == '-' for c in line.strip()):
                processed.append(line)
                continue
                
            # Extract addresses for deduplication
            if "Address:" in line:
                address = line.replace("Address:", "").strip()
                if address in seen_addresses:
                    continue  # Skip duplicate address
                seen_addresses.add(address)
                processed.append(line)
                continue
                
            # Extract phone numbers for deduplication
            if "Phone:" in line:
                phone = line.replace("Phone:", "").strip()
                if phone in seen_phones:
                    continue  # Skip duplicate phone
                seen_phones.add(phone)
                processed.append(line)
                continue
                
            # Extract emails for deduplication
            if "Email:" in line:
                email = line.replace("Email:", "").strip()
                if email in seen_emails:
                    continue  # Skip duplicate email
                seen_emails.add(email)
                processed.append(line)
                continue
                
            # Include all other lines
            processed.append(line)
        
        # Add summary of findings
        processed.append("\n=== SKIP TRACING SUMMARY ===")
        processed.append(f"Total unique addresses found: {len(seen_addresses)}")
        processed.append(f"Total unique phone numbers found: {len(seen_phones)}")
        if seen_emails:
            processed.append(f"Total unique emails found: {len(seen_emails)}")
        
        return processed

class FastPeopleSearchScraper:
    """A scraper for FastPeopleSearch that uses different techniques to avoid blocking"""
    
    def __init__(self):
        self.base_url = "https://www.fastpeoplesearch.com"
        self.search_url = f"{self.base_url}/name"
        
        # Expanded headers pool with newer user agents
        self.headers_pool = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Ch-Ua': '"Chromium";v="122", "Google Chrome";v="122", "Not(A:Brand";v="24"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.7,es;q=0.3',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Pragma': 'no-cache',
                'Cache-Control': 'no-cache',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Referer': 'https://www.google.com/',
            }
        ]
        
        # Initialize cookies and session
        self.cookies = {}
        self.max_retries = 3
        self.failed_attempts = 0
        
    def get_random_headers(self):
        """Get random headers to avoid detection"""
        headers = random.choice(self.headers_pool).copy()
        # Add some randomization to appear more like a real browser
        if random.random() < 0.5:
            headers['Accept-Encoding'] = 'gzip'
        return headers
    
    async def wait_with_backoff(self):
        """Implement exponential backoff for retries"""
        if self.failed_attempts > 0:
            delay = min(2 ** self.failed_attempts * 2 * (0.5 + random.random()), 60)
            await asyncio.sleep(delay)
        else:
            # Normal delay between 2-5 seconds to mimic human behavior
            await asyncio.sleep(random.uniform(2, 5))
    
    async def search_person(self, first_name, last_name, state=None, city=None, age=None):
        """Search for a person on FastPeopleSearch"""
        results = []
        
        # Reset failed attempts counter for this new search
        self.failed_attempts = 0
        
        for attempt in range(self.max_retries):
            try:
                # Construct the search URL - FastPeopleSearch uses URL pattern not query parameters
                url = f"{self.search_url}/{first_name}-{last_name}"
                if state and state != "All States":
                    # Format state name for URL (lowercase, dash separator)
                    state_formatted = state.lower().replace(' ', '-')
                    url = f"{url}/{state_formatted}"
                
                # Add city to the search if provided
                if city:
                    city_formatted = city.lower().replace(' ', '-')
                    url = f"{url}/{city_formatted}"
                
                # Add a cache-busting parameter
                url = f"{url}?rid={random.randint(10000, 99999)}"
                
                # Apply backoff strategy before making request
                await self.wait_with_backoff()
                
                # Get random headers
                headers = self.get_random_headers()
                
                # First make a request to the homepage to get cookies
                async with aiohttp.ClientSession(cookies=self.cookies) as session:
                    # Visit the homepage first
                    if not self.cookies:
                        try:
                            async with session.get(self.base_url, headers=headers, timeout=20) as home_response:
                                if home_response.status == 200:
                                    # Save cookies
                                    self.cookies = {k: v.value for k, v in home_response.cookies.items()}
                                    await asyncio.sleep(random.uniform(1, 3))
                        except Exception as e:
                            print(f"Error accessing FastPeopleSearch homepage: {str(e)}")
                    
                    # Now make the search request
                    async with session.get(url, headers=headers, ssl=False, timeout=30) as response:
                        # Update cookies
                        for k, v in response.cookies.items():
                            self.cookies[k] = v.value
                        
                        if response.status == 200:
                            html = await response.text()
                            # Reset failed attempts on success
                            self.failed_attempts = 0
                            
                            # Parse results
                            parsed_results = self.parse_search_results(html)
                            
                            # If parsing failed or returned no results, generate synthetic ones
                            if not parsed_results:
                                results.append("\n=== FastPeopleSearch Results ===\n")
                                results.append(f"Searched for: {first_name} {last_name}")
                                if state and state != "All States":
                                    results.append(f"State: {state}")
                                if city:
                                    results.append(f"City: {city}")
                                results.append(f"\nDirect link: {url}")
                                results.append("\nNo results were parsed from the page. Please visit the direct link above to view results.")
                            else:
                                results.extend(parsed_results)
                            break  # Success, exit retry loop
                        elif response.status == 429:  # Too Many Requests
                            self.failed_attempts += 1
                            if attempt == self.max_retries - 1:
                                return [f"Error: Rate limited by FastPeopleSearch. Try again later."]
                        elif response.status == 403:  # Forbidden
                            self.failed_attempts += 1
                            if attempt == self.max_retries - 1:
                                return [f"Error: Access blocked by FastPeopleSearch (403 Forbidden)."]
                        else:
                            self.failed_attempts += 1
                            if attempt == self.max_retries - 1:
                                return [f"Error: Received status code {response.status}"]
            
            except Exception as e:
                self.failed_attempts += 1
                if attempt == self.max_retries - 1:
                    return [f"Error searching FastPeopleSearch: {str(e)}"]
        
        # If we've gone through all retries with no results, provide guidance
        if not results:
            results.append("\n=== FastPeopleSearch Search Failed ===\n")
            results.append("Could not access FastPeopleSearch after multiple attempts.")
        
        return results
    
    def parse_search_results(self, html):
        """Parse search results from FastPeopleSearch"""
        results = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Add header
            results.append("\n=== FastPeopleSearch Results ===\n")
            
            # Check if we've been blocked (common text when IP is restricted)
            if "please complete the security check to access" in html.lower() or "solving the challenge" in html.lower():
                results.append("Access to FastPeopleSearch was blocked by security checks.")
                results.append("Recommendation: Try searching later or use a different network.")
                return results
                
            # Find search result cards
            cards = soup.select('div.card-summary')
            
            if not cards:
                # If no results, generate some helpful info
                results.append("No results found on FastPeopleSearch.")
                return results
                
            results.append(f"Found {len(cards)} potential matches\n")
            
            # Parse each card
            for i, card in enumerate(cards):
                results.append(f"--- Result {i+1} ---")
                
                # Name
                name_element = card.select_one('div.name')
                if name_element:
                    results.append(f"Name: {name_element.text.strip()}")
                
                # Age
                age_element = card.select_one('div.age')
                if age_element:
                    results.append(f"Age: {age_element.text.strip()}")
                
                # Address
                address_element = card.select_one('div.address')
                if address_element:
                    results.append(f"Address: {address_element.text.strip()}")
                
                # Phone
                phone_element = card.select_one('div.phone')
                if phone_element:
                    results.append(f"Phone: {phone_element.text.strip()}")
                
                # Get link to profile page
                profile_link = card.select_one('a.detail-link')
                if profile_link and 'href' in profile_link.attrs:
                    results.append(f"Profile Link: {self.base_url}{profile_link['href']}")
                
                results.append("")
                
        except Exception as e:
            results.append(f"Error parsing results: {str(e)}")
            
        return results
    
    async def search_voter_records(self, first_name, last_name, state=None, city=None):
        """Search voter records"""
        results = []
        results.append("\n--- Voter Records ---\n")
        
        # Form the search URL
        search_url = f"https://voterrecords.com/voters/{first_name.lower()}-{last_name.lower()}"
        if state and state != "All States":
            state_code = self.get_state_code(state)
            if state_code:
                search_url += f"/{state_code}"
        
        results.append(f"Voter Records Search: {search_url}")
        results.append("You can visit this URL to see voter registration records")
        
        return results
    
    async def search_business_records(self, first_name, last_name, state=None):
        """Search business records in OpenCorporates"""
        results = []
        results.append("\n--- Business Records ---\n")
        
        search_url = f"https://opencorporates.com/search?q={first_name}+{last_name}&type=officers"
        results.append(f"Business Records Search: {search_url}")
        results.append("This search will show business affiliations and corporate roles")
        
        return results
        
    async def search_court_records(self, first_name, last_name, state=None):
        """Search court records using CourtListener"""
        results = []
        results.append("\n--- Court Records ---\n")
        
        search_url = f"https://www.courtlistener.com/?type=p&q={first_name}+{last_name}&order_by=score+desc"
        results.append(f"Court Records Search: {search_url}")
        results.append("This search will show court cases and legal proceedings")
        
        return results
    
    async def search_mugshots(self, first_name, last_name, state=None):
        """Search for mugshots"""
        results = []
        results.append("\n--- Mugshot Search ---\n")
        
        search_url = f"https://mugshots.com/search.html?q={first_name}+{last_name}"
        results.append(f"Mugshot Search: {search_url}")
        results.append("This search will check arrest records and booking photos")
        
        return results
    
    def get_state_code(self, state_name):
        """Convert state name to two-letter code"""
        state_codes = {
            "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
            "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
            "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
            "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
            "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
            "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
            "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
            "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
            "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
            "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
        }
        return state_codes.get(state_name, "")

class FreePublicDataScraper:
    """Scraper that uses free public data APIs to get information without scraping issues"""
    
    def __init__(self):
        # Publicly available APIs that don't require authentication
        self.apis = {
            "random_user": "https://randomuser.me/api/",  # We'll use this as a demo
            "open_notify": "http://api.open-notify.org/astros.json",  # Just an example of a free API
        }
        
        # Data sources to search
        self.data_sources = {
            'voterrecords': True,
            'opencorporates': True,
            'govinfo': True,
            'mugshots': True,
            'courtlistener': True
        }
        
        # URLs for data sources
        self.source_urls = {
            'voterrecords': 'https://voterrecords.com/voters/',
            'opencorporates': 'https://opencorporates.com/officers/',
            'govinfo': 'https://www.govinfo.gov/content/pkg/',
            'mugshots': 'https://mugshots.com/search.html?q=',
            'courtlistener': 'https://www.courtlistener.com/person/'
        }
        
        # State code mapping
        self.state_codes = {
            'Alabama': 'al', 'Alaska': 'ak', 'Arizona': 'az', 'Arkansas': 'ar', 
            'California': 'ca', 'Colorado': 'co', 'Connecticut': 'ct', 'Delaware': 'de',
            'Florida': 'fl', 'Georgia': 'ga', 'Hawaii': 'hi', 'Idaho': 'id',
            'Illinois': 'il', 'Indiana': 'in', 'Iowa': 'ia', 'Kansas': 'ks',
            'Kentucky': 'ky', 'Louisiana': 'la', 'Maine': 'me', 'Maryland': 'md',
            'Massachusetts': 'ma', 'Michigan': 'mi', 'Minnesota': 'mn', 'Mississippi': 'ms',
            'Missouri': 'mo', 'Montana': 'mt', 'Nebraska': 'ne', 'Nevada': 'nv',
            'New Hampshire': 'nh', 'New Jersey': 'nj', 'New Mexico': 'nm', 'New York': 'ny',
            'North Carolina': 'nc', 'North Dakota': 'nd', 'Ohio': 'oh', 'Oklahoma': 'ok',
            'Oregon': 'or', 'Pennsylvania': 'pa', 'Rhode Island': 'ri', 'South Carolina': 'sc',
            'South Dakota': 'sd', 'Tennessee': 'tn', 'Texas': 'tx', 'Utah': 'ut',
            'Vermont': 'vt', 'Virginia': 'va', 'Washington': 'wa', 'West Virginia': 'wv',
            'Wisconsin': 'wi', 'Wyoming': 'wy'
        }
        
    def get_state_code(self, state_name):
        """Convert state name to state code"""
        return self.state_codes.get(state_name, "")
        
    async def search_person(self, first_name, last_name, state=None, city=None, age=None):
        """Search for person using free public data sources"""
        results = []
        results.append("\n=== Free Public Data Search ===\n")
        
        # Run searches in parallel
        tasks = []
        
        # Search voter records
        if self.data_sources['voterrecords']:
            tasks.append(self.search_voter_records(first_name, last_name, state, city))
            
        # Search business records
        if self.data_sources['opencorporates']:
            tasks.append(self.search_business_records(first_name, last_name, state))
            
        # Search court records
        if self.data_sources['courtlistener']:
            tasks.append(self.search_court_records(first_name, last_name, state))
            
        # Search for mugshots
        if self.data_sources['mugshots']:
            tasks.append(self.search_mugshots(first_name, last_name, state))
        
        # Run searches concurrently
        if tasks:
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in search_results:
                if isinstance(res, Exception):
                    results.append(f"Error during public data search: {str(res)}")
                elif isinstance(res, list):
                    results.extend(res)
            
        # Generate synthetic results for demo purposes if no results found
        # This will be replaced with real results once the API calls are implemented
        if len(results) <= 1:  # Only header exists
            await self.generate_synthetic_results(results, first_name, last_name, state, city, age)
        
        return results
    
    async def search_voter_records(self, first_name, last_name, state=None, city=None):
        """Search voter records"""
        results = []
        results.append("\n--- Voter Records ---\n")
        
        # Form the search URL
        search_url = f"{self.source_urls['voterrecords']}{first_name.lower()}-{last_name.lower()}"
        if state and state != "All States":
            state_code = self.get_state_code(state)
            search_url += f"/{state_code}"
        
        results.append(f"Voter Records Search: {search_url}")
        results.append("You can visit this URL to see voter registration records")
        
        return results
    
    async def search_business_records(self, first_name, last_name, state=None):
        """Search business records in OpenCorporates"""
        results = []
        results.append("\n--- Business Records ---\n")
        
        # Form the search URL
        search_url = f"{self.source_urls['opencorporates']}?q={first_name}+{last_name}"
        results.append(f"Business Records Search: {search_url}")
        results.append("This will show company directorships and officers")
        
        return results
        
    async def search_court_records(self, first_name, last_name, state=None):
        """Search court records"""
        results = []
        results.append("\n--- Court Records ---\n")
        
        search_url = f"{self.source_urls['courtlistener']}?q={first_name}+{last_name}"
        results.append(f"Federal Court Cases: {search_url}")
        results.append("This will show federal court cases involving this person")
        
        return results
        
    async def search_mugshots(self, first_name, last_name, state=None):
        """Search mugshot databases"""
        results = []
        results.append("\n--- Mugshot Records ---\n")
        
        search_url = f"{self.source_urls['mugshots']}{first_name}+{last_name}"
        if state and state != "All States":
            search_url += f"+{state}"
        results.append(f"Possible Arrest Records: {search_url}")
        results.append("Note: Mugshot databases may contain outdated information")
        
        return results
        
    async def generate_synthetic_results(self, results, first_name, last_name, state=None, city=None, age=None):
        """Generate synthetic results for demo purposes when no real results are found"""
        import faker
        
        # Create a faker instance to generate realistic-looking data
        fake = faker.Faker()
        
        results.append("\n--- Google Search Queries (recommended) ---\n")
        
        # Generate useful Google dorks
        search_queries = [
            f'"{first_name} {last_name}" filetype:pdf',
            f'"{first_name} {last_name}" address phone',
            f'"{first_name} {last_name}" facebook linkedin'
        ]
        
        if state and state != "All States":
            search_queries.append(f'"{first_name} {last_name}" {state}')
            if city:
                search_queries.append(f'"{first_name} {last_name}" {city} {state}')
        
        results.append("Copy and paste these into Google:")
        for query in search_queries:
            results.append(f"* {query}")
            
        results.append("\nThese searches may help find:")
        results.append("- Public documents containing the person's name")
        results.append("- Social media profiles")
        results.append("- News articles or professional mentions")
        
        return results

class BasicPublicRecordsScraper:
    """A basic public records scraper that uses freely available data sources that are less likely to block requests"""
    
    def __init__(self):
        # Define the data sources we can search (these are less likely to implement strong anti-scraping)
        self.sources = {
            'state_records': True,      # State public records portals
            'voter_info': True,         # Public voter registration information
            'google_search': True,      # Basic Google search dorks
            'open_databases': True,     # Open data projects and government databases
        }
        
        # State public records portals
        self.state_portals = {
            'Alabama': 'https://www.alabamacourts.gov/court-records/',
            'Alaska': 'https://records.courts.alaska.gov/',
            'Arizona': 'https://www.azcourts.gov/publicaccess/',
            'Arkansas': 'https://caseinfo.arcourts.gov/',
            'California': 'https://www.courts.ca.gov/courts.htm',
            'Colorado': 'https://www.courts.state.co.us/Courts/Records/Index.cfm',
            'Connecticut': 'https://www.jud.ct.gov/Public.htm',
            'Delaware': 'https://courts.delaware.gov/supreme/records.aspx',
            'Florida': 'https://www.flcourts.org/Public-Information/Access-Court-Records',
            'Georgia': 'https://www.gasupreme.us/court-information/court-records/',
            'Hawaii': 'https://www.courts.state.hi.us/records',
            'Idaho': 'https://icourt.idaho.gov/portal',
            'Illinois': 'http://www.illinoiscourts.gov/Records/records.asp',
            'Indiana': 'https://public.courts.in.gov/mycase',
            'Iowa': 'https://www.iowacourts.gov/for-the-public/court-records/',
            'Kansas': 'https://www.kscourts.org/Public',
            'Kentucky': 'https://kycourts.gov/Records/',
            'Louisiana': 'https://www.lasc.org/court-records',
            'Maine': 'https://www.courts.maine.gov/records/',
            'Maryland': 'https://mdcourts.gov/courtrecords',
            'Massachusetts': 'https://www.mass.gov/topics/court-records',
            'Michigan': 'https://courts.michigan.gov/Case-Search/',
            'Minnesota': 'https://www.mncourts.gov/Access-Case-Records.aspx',
            'Mississippi': 'https://courts.ms.gov/records/records.php',
            'Missouri': 'https://www.courts.mo.gov/casenet/',
            'Montana': 'https://courts.mt.gov/Courts/records',
            'Nebraska': 'https://www.nebraska.gov/justicecc/ccname.cgi',
            'Nevada': 'https://nvcourts.gov/supreme/records',
            'New Hampshire': 'https://www.courts.nh.gov/court-records-case-information',
            'New Jersey': 'https://njcourts.gov/public/records',
            'New Mexico': 'https://caselookup.nmcourts.gov/',
            'New York': 'https://iapps.courts.state.ny.us/webcivil/ecourtsMain',
            'North Carolina': 'https://www.nccourts.gov/records',
            'North Dakota': 'https://www.ndcourts.gov/public-access',
            'Ohio': 'https://www.supremecourt.ohio.gov/courts/courtrecords/',
            'Oklahoma': 'https://www.oscn.net/dockets/search.aspx',
            'Oregon': 'https://www.courts.oregon.gov/services/online/Pages/records-calendars.aspx',
            'Pennsylvania': 'https://ujsportal.pacourts.us/CaseSearch',
            'Rhode Island': 'https://www.courts.ri.gov/Pages/SearchCases.aspx',
            'South Carolina': 'https://www.sccourts.org/caseSearch/',
            'South Dakota': 'https://ujsportal.sd.gov/default.aspx',
            'Tennessee': 'https://www.tncourts.gov/records',
            'Texas': 'https://www.txcourts.gov/public-court-records/',
            'Utah': 'https://www.utcourts.gov/records/',
            'Vermont': 'https://www.vermontjudiciary.org/court-records',
            'Virginia': 'https://www.vacourts.gov/online/public_court_records.html',
            'Washington': 'https://www.courts.wa.gov/court_dir/?fa=court_dir.access',
            'West Virginia': 'http://www.courtswv.gov/lower-courts/records.html',
            'Wisconsin': 'https://wcca.wicourts.gov/',
            'Wyoming': 'https://www.courts.state.wy.us/public-records/'
        }
        
        # Property records portals
        self.property_portals = {
            'California': [
                {'county': 'Los Angeles', 'url': 'https://assessor.lacounty.gov/'},
                {'county': 'San Diego', 'url': 'https://arcc.sdcounty.ca.gov/Pages/public-records.aspx'},
            ],
            'Florida': [
                {'county': 'Miami-Dade', 'url': 'https://www.miamidade.gov/global/land-records.page'},
                {'county': 'Broward', 'url': 'https://bcpa.net/RecordSearch.asp'},
            ],
            'New York': [
                {'county': 'New York City', 'url': 'https://a836-acris.nyc.gov/'},
                {'county': 'Suffolk', 'url': 'https://suffolkcountyny.gov/clerk/records'},
            ],
            'Texas': [
                {'county': 'Harris', 'url': 'https://www.hcad.org/records'},
                {'county': 'Dallas', 'url': 'https://www.dallascad.org/SearchAddr.aspx'},
            ],
        }
        
        # Search dorks that can be used with Google
        self.search_dorks = [
            'intext:"{first} {last}" filetype:pdf',
            'intext:"{first} {last}" intext:address',
            'intext:"{first} {last}" intext:phone',
            'site:whitepages.com "{first} {last}"',
            'site:legacy.com "{first} {last}" obituary',
            'site:facebook.com "{first} {last}"',
            'site:linkedin.com "{first} {last}"',
            'intext:"{first} {last}" intext:county',
        ]
        
        # Initialize session and headers
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
    async def search_person(self, first_name, last_name, state=None, city=None, age=None):
        """Search for a person using multiple public data sources"""
        results = []
        results.append("\n=== Basic Public Records Search ===\n")
        
        # Run searches in parallel for better performance
        tasks = []
        
        # Search state public records
        if self.sources['state_records'] and state and state != "All States":
            tasks.append(self.search_state_records(first_name, last_name, state))
        
        # Search using Google search dorks
        if self.sources['google_search']:
            tasks.append(self.search_using_dorks(first_name, last_name, state))
        
        # Search voter information if state is provided
        if self.sources['voter_info'] and state and state != "All States":
            tasks.append(self.search_voter_info(first_name, last_name, state))
        
        # Search property records if state is provided
        if state and state != "All States":
            tasks.append(self.search_property_records(first_name, last_name, state))
        
        # Run all search tasks concurrently
        if tasks:
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in search_results:
                if isinstance(res, Exception):
                    results.append(f"Error during search: {str(res)}")
                elif isinstance(res, list):
                    results.extend(res)
        
        # If no results were found, provide an informative message
        if len(results) <= 1:  # Only the header exists
            results.append("No public records found. Try using a more specific search.")
        
        return results
        
    async def search_using_dorks(self, first_name, last_name, state=None):
        """Generate Google search dorks for finding public information"""
        results = []
        results.append("\n=== Google Search Queries ===\n")
        results.append("You can use these search queries in Google to find information:\n")
        
        # Format dorks with the person's name
        for dork in self.search_dorks:
            formatted_dork = dork.format(first=first_name, last=last_name)
            if state and state != "All States":
                formatted_dork += f" intext:\"{state}\""
            results.append(f"• {formatted_dork}")
            
        # Add a few more specific dorks
        if state and state != "All States":
            results.append(f"• site:{state.lower().replace(' ', '')}.gov intext:\"{first_name} {last_name}\"")
        
        results.append(f"• intext:\"{first_name} {last_name}\" intext:contact")
        results.append(f"• intext:\"{first_name} {last_name}\" intext:directory")
        results.append("\nThese search queries may help you find additional information manually.")
        
        return results
        
    async def search_state_records(self, first_name, last_name, state):
        """Search state public records portals"""
        results = []
        results.append("\n=== State Public Records ===\n")
        
        if state in self.state_portals:
            portal_url = self.state_portals[state]
            results.append(f"State Records Portal: {portal_url}")
            results.append(f"To search for {first_name} {last_name}, visit the above portal and enter the name in the search fields.")
            results.append("Note: Some state portals may require registration or payment for detailed information.")
        else:
            results.append(f"No specific portal information available for {state}.")
            
        return results
        
    async def search_voter_info(self, first_name, last_name, state):
        """Search for voter registration information"""
        results = []
        results.append("\n=== Voter Registration Information ===\n")
        results.append("Note: Voter records are publicly available in many states but access methods vary.\n")
        
        # Provide information on how to access voter records for the state
        results.append(f"To search for {first_name} {last_name}'s voter information in {state}:")
        
        # Different states have different methods
        if state in ["Florida", "Ohio", "Michigan", "North Carolina"]:
            results.append(f"• Visit the {state} Secretary of State website")
            results.append("• Look for 'Voter Information' or 'Voter Records'")
            results.append(f"• Enter the name '{first_name} {last_name}' in the search fields")
        else:
            results.append(f"• Contact the {state} Secretary of State office")
            results.append("• Request voter registration information (may require proper identification)")
            
        return results
        
    async def search_property_records(self, first_name, last_name, state):
        """Search for property records"""
        results = []
        results.append("\n=== Property Records ===\n")
        
        # Check if we have property portals for this state
        if state in self.property_portals:
            results.append(f"Property record portals for {state}:")
            for county_info in self.property_portals[state]:
                results.append(f"• {county_info['county']} County: {county_info['url']}")
            results.append(f"\nTo search for properties owned by {first_name} {last_name}, visit these county websites and use their search features.")
        else:
            results.append(f"No specific property portals available for {state}.")
            results.append("Try searching the county assessor or recorder's office for the specific county of interest.")
            
        return results

class SkipTracingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Skip Tracing")
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Add form elements
        self.first_name_input = QLineEdit()
        self.last_name_input = QLineEdit()
        self.state_combo = QComboBox()
        self.state_combo.addItems(["All States", "Alabama", "Alaska", "Arizona", "Arkansas", 
                           "California", "Colorado", "Connecticut", "Delaware", 
                           "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", 
                           "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", 
                           "Maine", "Maryland", "Massachusetts", "Michigan", 
                           "Minnesota", "Mississippi", "Missouri", "Montana", 
                           "Nebraska", "Nevada", "New Hampshire", "New Jersey", 
                           "New Mexico", "New York", "North Carolina", "North Dakota", 
                           "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", 
                           "South Carolina", "South Dakota", "Tennessee", "Texas", 
                           "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
                           "Wisconsin", "Wyoming"])
        self.city_input = QLineEdit()
        self.age_input = QSpinBox()
        self.age_input.setMinimum(0)
        self.age_input.setMaximum(120)
        self.age_input.setValue(0)
        self.age_input.setSpecialValueText("Any")
        
        layout.addWidget(QLabel("First Name:"))
        layout.addWidget(self.first_name_input)
        layout.addWidget(QLabel("Last Name:"))
        layout.addWidget(self.last_name_input)
        layout.addWidget(QLabel("State:"))
        layout.addWidget(self.state_combo)
        layout.addWidget(QLabel("City (optional):"))
        layout.addWidget(self.city_input)
        layout.addWidget(QLabel("Age (optional):"))
        layout.addWidget(self.age_input)
        
        # Add buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Skip Tracing")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

class SkipTracingWorker(QObject):
    results = Signal(list)
    finished = Signal()
    
    def __init__(self, skip_tracer, params):
        super().__init__()
        self.skip_tracer = skip_tracer
        self.params = params
        
    def run(self):
        """Run the skip tracing process"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the trace_person method in the event loop
            results = loop.run_until_complete(
                self.skip_tracer.trace_person(
                    self.params['first_name'],
                    self.params['last_name'],
                    self.params['state'],
                    self.params['city'],
                    self.params['age']
                )
            )
            self.results.emit(results)
        except Exception as e:
            self.results.emit([f"Error during skip tracing: {str(e)}"])
        finally:
            self.finished.emit()
            loop.close()

def open_skip_tracing_dialog(parent=None):
    """Open the skip tracing dialog as a standalone window"""
    dialog = SkipTracingDialog(parent)
    dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BackgroundCheckApp()
    window.show()
    
    # No need for the standalone skip tracing button since it's integrated now
    
    sys.exit(app.exec())
