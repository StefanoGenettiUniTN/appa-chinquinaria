#!/usr/bin/env python3
"""
Script to download files from URLs listed in a CSV file.
This script is designed to work with EEA ParquetFilesUrls CSV files.
"""

import os
import csv
import requests
import argparse
import time
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs
import hashlib
from pathlib import Path

def clean_filename(url):
    """
    Generate a clean filename from URL, removing query parameters and special characters.
    """
    # Remove query parameters
    clean_url = url.split("?")[0]
    # Get the filename from the URL
    filename = os.path.basename(clean_url)
    
    # If no filename in URL, generate one from URL hash
    if not filename or '.' not in filename:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"file_{url_hash}.parquet"
    
    return filename

def download_file(url, download_dir, timeout=60, max_retries=3):
    """
    Download a single file with retry logic.
    
    Args:
        url (str): URL to download
        download_dir (str): Directory to save the file
        timeout (int): Request timeout in seconds
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        bool: True if successful, False otherwise
    """
    filename = clean_filename(url)
    filepath = os.path.join(download_dir, filename)
    
    # Skip if file already exists
    if os.path.exists(filepath):
        return True
    
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                
                # Get file size for progress bar
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
                # Verify file was downloaded completely
                if total_size > 0 and downloaded != total_size:
                    print(f"⚠️  Warning: {filename} - Expected {total_size} bytes, got {downloaded}")
                    return False
                
                return True
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Attempt {attempt + 1}/{max_retries} failed for {filename}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return False
        except Exception as e:
            print(f"❌ Unexpected error downloading {filename}: {e}")
            return False
    
    return False

def read_urls_from_csv(csv_path):
    """
    Read URLs from CSV file.
    Supports multiple CSV formats:
    - Single URL per row
    - Multiple columns with URL in first column
    - Headers that should be skipped
    - BOM-encoded files
    """
    urls = []
    
    # Try different encodings to handle BOM
    encodings = ['utf-8-sig', 'utf-8', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as csvfile:
                # Read first few lines to understand the format
                first_lines = []
                for i, line in enumerate(csvfile):
                    if i >= 5:  # Read first 5 lines
                        break
                    first_lines.append(line.strip())
                
                csvfile.seek(0)
                
                # Check if it's a simple line-separated format (no commas)
                if len(first_lines) > 1 and ',' not in first_lines[1]:
                    # Simple line-separated format
                    for line_num, line in enumerate(csvfile, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Skip header if it doesn't look like a URL
                        if line_num == 1 and not line.startswith("http"):
                            continue
                            
                        if line.startswith("http"):
                            urls.append(line)
                else:
                    # Traditional CSV format
                    # Try to detect delimiter
                    sample = csvfile.read(1024)
                    csvfile.seek(0)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    
                    reader = csv.reader(csvfile, delimiter=delimiter)
                    
                    for row_num, row in enumerate(reader, 1):
                        if not row:  # Skip empty rows
                            continue
                        
                        # Look for URL in any column
                        for col in row:
                            if col and col.strip().startswith("http"):
                                urls.append(col.strip())
                                break
                        else:
                            # If no URL found in row, check if it might be a header
                            if row_num == 1 and any(header in str(row).lower() for header in ['url', 'link', 'file']):
                                continue  # Skip header row
            
            # If we successfully read URLs, break out of encoding loop
            if urls:
                break
                
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"Error reading CSV with encoding {encoding}: {e}")
            continue
    
    return urls

def main():
    parser = argparse.ArgumentParser(
        description="Download files from URLs listed in a CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_from_csv.py --csv ParquetFilesUrls.csv --output downloads
  python download_from_csv.py --csv urls.csv --output ./data --timeout 120 --retries 5
        """
    )
    
    parser.add_argument('--csv', required=True, help='Path to CSV file containing URLs')
    parser.add_argument('--output', required=False, help='Directory to save downloaded files (default: data/eea-downloads)')
    parser.add_argument('--timeout', type=int, default=60, help='Request timeout in seconds (default: 60)')
    parser.add_argument('--retries', type=int, default=3, help='Maximum retry attempts (default: 3)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip files that already exist')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv):
        print(f"❌ Error: CSV file '{args.csv}' not found")
        return 1
    
    # Set default output directory if not provided
    if args.output is None:
        # Always save in data/eea-downloads folder in project root
        project_root = Path(__file__).parent.parent.parent
        args.output = project_root / "data" / "eea-downloads"
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"📁 Output directory: {args.output}")
    
    # Read URLs from CSV
    print(f"📖 Reading URLs from: {args.csv}")
    urls = read_urls_from_csv(args.csv)
    
    if not urls:
        print("❌ No URLs found in CSV file")
        return 1
    
    print(f"🔗 Found {len(urls)} URLs to download")
    
    # Download files
    successful_downloads = 0
    failed_downloads = 0
    skipped_files = 0
    
    for url in tqdm(urls, desc="Downloading files"):
        filename = clean_filename(url)
        filepath = os.path.join(args.output, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            if args.skip_existing:
                skipped_files += 1
                continue
            else:
                # Remove existing file to re-download
                os.remove(filepath)
        
        # Download the file
        if download_file(url, args.output, args.timeout, args.retries):
            successful_downloads += 1
        else:
            failed_downloads += 1
    
    # Summary
    print("\n" + "="*50)
    print("📊 DOWNLOAD SUMMARY")
    print("="*50)
    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"⏭️  Skipped files: {skipped_files}")
    print(f"📁 Total files in directory: {len(os.listdir(args.output))}")
    
    if failed_downloads > 0:
        print(f"\n⚠️  {failed_downloads} files failed to download. Check the output above for details.")
        return 1
    else:
        print("\n🎉 All downloads completed successfully!")
        return 0

if __name__ == "__main__":
    exit(main())
