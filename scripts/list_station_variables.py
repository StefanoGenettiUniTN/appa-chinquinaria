#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
List all available variables for a specific station
"""

import sys
import requests
import xml.etree.ElementTree as ET

def list_variables(station_code):
    """List all variables for a station."""
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
    url = f"{base_url}/wgen/cache/anon/cf{station_code}.xml"
    
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        
        print(f"Available variables for station {station_code}:")
        print("=" * 60)
        
        for var in root.findall('.//variable'):
            name = var.attrib.get('name', '')
            subdesc = var.attrib.get('subdesc', '')
            var_id = var.attrib.get('var', '')
            unit = var.attrib.get('varunits', '')
            period = var.attrib.get('varperiod', '')
            
            # Mark if it's an "Annale" variant
            annale_mark = " (ANNALE VARIANT)" if "Annale" in subdesc else ""
            
            print(f"ID: {var_id}")
            print(f"Name: {name}{annale_mark}")
            print(f"Unit: {unit}")
            print(f"Description: {subdesc}")
            print(f"Period: {period}")
            print("-" * 40)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python list_station_variables.py <STATION_CODE>")
        print("Example: python list_station_variables.py T0038")
        sys.exit(1)
    
    station_code = sys.argv[1]
    list_variables(station_code)
