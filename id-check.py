import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLineEdit, QPushButton, QLabel, 
                              QTextEdit, QComboBox, QTabWidget, QProgressBar, 
                              QStatusBar, QSpinBox, QCheckBox, QFileDialog, 
                              QMessageBox, QDialog, QGroupBox, QDateEdit, 
                              QGridLayout)
from PySide6.QtCore import Qt, QThread, Signal
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
        self.sources = {
            'social': self.search_social_media,
            'public': self.search_public_records,
            'dark': self.search_dark_web,
            'github': self.search_github,
            'leaks': self.search_data_leaks
        }
        self.rate_limiters = {}
        self.setup_rate_limits()

    def setup_rate_limits(self):
        """Setup rate limits for different APIs"""
        self.rate_limiters = {
            'github': sleep_and_retry(limits(calls=30, period=60)),
            'social': sleep_and_retry(limits(calls=100, period=60)),
            'public': sleep_and_retry(limits(calls=50, period=60))
        }

    async def search_social_media(self, params):
        """Search social media platforms"""
        results = []
        try:
            # LinkedIn search
            linkedin_results = await self.search_linkedin(params)
            results.extend(linkedin_results)
            
            # Twitter search
            twitter_results = await self.search_twitter(params)
            results.extend(twitter_results)
            
            return results
        except Exception as e:
            logging.error(f"Social media search error: {str(e)}")
            return []

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

    def set_search_params(self, params):
        self.search_params = params

    async def run_search(self):
        """Run the comprehensive search"""
        all_results = []
        total_sources = len(self.orchestrator.sources)
        completed = 0

        for source_name, search_func in self.orchestrator.sources.items():
            if not self.is_running:
                break

            self.status.emit(f"Searching {source_name}...")
            try:
                results = await search_func(self.search_params)
                all_results.extend(results)
            except Exception as e:
                logging.error(f"Error searching {source_name}: {str(e)}")

            completed += 1
            self.progress.emit(int((completed / total_sources) * 100))

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
        options_layout.addWidget(self.use_github)
        options_layout.addWidget(self.use_social)
        
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BackgroundCheckApp()
    window.show()
    sys.exit(app.exec())
