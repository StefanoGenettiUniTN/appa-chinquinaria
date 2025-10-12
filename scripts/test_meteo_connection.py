#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify Meteo Trentino connectivity and timeouts
"""

import requests
import time
from bs4 import BeautifulSoup

def test_connection():
    """Test basic connectivity to Meteo Trentino."""
    session = requests.Session()
    session.cookies.update({
        "bandwidth": "high",
        "username": "webuser",
        "userclass": "anon",
        "is_admin": "0",
        "fontsize": "80.01",
        "plotsize": "normal",
        "menuwidth": "20"
    })
    
    base_url = "http://storico.meteotrentino.it"
    
    print("Testing Meteo Trentino connectivity...")
    
    # Test 1: Basic connectivity
    try:
        print("1. Testing basic connectivity...")
        response = session.get(base_url, timeout=(5, 10))
        print(f"   Status: {response.status_code}")
        print(f"   Response time: {response.elapsed.total_seconds():.2f}s")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 2: Variables endpoint
    try:
        print("2. Testing variables endpoint...")
        url = f"{base_url}/wgen/cache/anon/cfT0038.xml"
        response = session.get(url, timeout=(5, 15))
        print(f"   Status: {response.status_code}")
        print(f"   Response time: {response.elapsed.total_seconds():.2f}s")
        print(f"   Content length: {len(response.text)} chars")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 3: Download trigger
    try:
        print("3. Testing download trigger...")
        params = {
            "co": "T0038",
            "v": "10.00_10.00",
            "vn": "Pioggia ",
            "p": "Tutti i dati,01/01/1800,01/01/1800,period,1",
            "o": "Download,download",
            "i": "Tutte le misure,Point,1",
            "cat": "rs"
        }
        response = session.get(f"{base_url}/cgi/webhyd.pl", params=params, timeout=(5, 120))
        print(f"   Status: {response.status_code}")
        print(f"   Response time: {response.elapsed.total_seconds():.2f}s")
        print(f"   Content length: {len(response.text)} chars")
        
        # Check for ZIP link
        soup = BeautifulSoup(response.text, 'html.parser')
        zip_found = False
        for script in soup.find_all("script"):
            if "downloadlink" in script.text:
                zip_found = True
                break
        print(f"   ZIP link found: {zip_found}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_connection()
