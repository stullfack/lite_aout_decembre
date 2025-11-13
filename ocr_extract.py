#!/usr/bin/env python3
"""
OCR extraction script for extracting date headings and names from screenshot images.

This script:
1. Uses GitHub Git Trees API to discover image files in the repository
2. Downloads raw image bytes from raw.githubusercontent.com
3. Preprocesses images (grayscale, optional resize) for better OCR
4. Runs Tesseract OCR with French+English languages
5. Parses OCR text to extract dates and names
6. Cleans, deduplicates, and formats the data
7. Writes CSV output with columns JJ.MM and Prénoms
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from io import BytesIO

import requests
from PIL import Image
import pytesseract


# French month names to numbers mapping
FRENCH_MONTHS = {
    'janvier': '01', 'janv': '01', 'jan': '01',
    'février': '02', 'fevrier': '02', 'fév': '02', 'fev': '02', 'feb': '02',
    'mars': '03', 'mar': '03',
    'avril': '04', 'avr': '04', 'apr': '04',
    'mai': '05', 'may': '05',
    'juin': '06', 'jun': '06',
    'juillet': '07', 'juil': '07', 'jul': '07',
    'août': '08', 'aout': '08', 'aoû': '08', 'aug': '08',
    'septembre': '09', 'sept': '09', 'sep': '09',
    'octobre': '10', 'oct': '10',
    'novembre': '11', 'nov': '11',
    'décembre': '12', 'decembre': '12', 'déc': '12', 'dec': '12',
}

# UI words to filter out from names
UI_WORDS = {
    'mes fêtes', 'mes fetes', 'favoris', 'ajouter', 'fêtes à souhaiter',
    'fetes à souhaiter', 'fêtes a souhaiter', 'fetes a souhaiter',
    'rechercher', 'recherche', 'lite', 'screenshot', 'souhaiter'
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract dates and names from screenshots using OCR'
    )
    parser.add_argument('--owner', required=True, help='GitHub repository owner')
    parser.add_argument('--repo', required=True, help='GitHub repository name')
    parser.add_argument('--branch', default='main', help='Branch name (default: main)')
    parser.add_argument('--output', default='fetes_extracted.csv',
                        help='Output CSV file (default: fetes_extracted.csv)')
    parser.add_argument('--token', default=None,
                        help='GitHub token (default: from GITHUB_TOKEN env var)')
    parser.add_argument('--local-path', default=None,
                        help='Local path to scan instead of using GitHub API (for testing)')
    return parser.parse_args()


def get_image_files(owner, repo, branch, token=None, local_path=None):
    """
    Use GitHub Git Trees API to list all image files in the repository.
    If local_path is provided, scan local filesystem instead.
    
    Returns a list of file paths for images (jpg, jpeg, png, gif).
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    
    # If local path is provided, scan local filesystem
    if local_path:
        print(f"Scanning local directory: {local_path}")
        image_files = []
        for root, dirs, files in os.walk(local_path):
            for file in files:
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    # Get relative path from local_path
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, local_path)
                    image_files.append(rel_path)
        print(f"Found {len(image_files)} image files")
        return image_files
    
    # Otherwise use GitHub API
    url = f'https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1'
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    print(f"Fetching repository tree from {url}...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    tree = response.json()
    image_files = []
    
    for item in tree.get('tree', []):
        if item['type'] == 'blob':
            path = item['path']
            _, ext = os.path.splitext(path.lower())
            if ext in image_extensions:
                image_files.append(path)
    
    print(f"Found {len(image_files)} image files")
    return image_files


def download_image(owner, repo, branch, path, local_path=None):
    """
    Download raw image bytes from raw.githubusercontent.com.
    If local_path is provided, read from local filesystem instead.
    
    Returns PIL Image object.
    """
    if local_path:
        # Read from local filesystem
        full_path = os.path.join(local_path, path)
        print(f"Reading {path}...")
        image = Image.open(full_path)
        return image
    
    # Download from GitHub
    url = f'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}'
    print(f"Downloading {path}...")
    
    response = requests.get(url)
    response.raise_for_status()
    
    image = Image.open(BytesIO(response.content))
    return image


def preprocess_image(image, min_width=800):
    """
    Preprocess image for better OCR accuracy.
    
    - Convert to grayscale
    - Optionally resize if smaller than threshold
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize if image is too small
    if image.width < min_width:
        scale_factor = min_width / image.width
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        image = image.resize(new_size, Image.LANCZOS)
    
    return image


def parse_date(text):
    """
    Parse a date heading from text.
    
    Supports formats like:
    - '26 Juillet'
    - '26.07'
    - '26 Juil'
    - '26/07'
    
    Returns normalized date as 'DD.MM' or None if not a valid date.
    """
    text = text.strip().lower()
    
    # Pattern 1: DD Month (e.g., "26 juillet", "26 juil")
    match = re.match(r'^(\d{1,2})\s+([a-zàéèêë]+)', text)
    if match:
        day = match.group(1).zfill(2)
        month_name = match.group(2)
        for key, month_num in FRENCH_MONTHS.items():
            if month_name.startswith(key) or key.startswith(month_name):
                return f"{day}.{month_num}"
    
    # Pattern 2: DD.MM or DD/MM (e.g., "26.07", "26/07")
    match = re.match(r'^(\d{1,2})[./](\d{1,2})$', text)
    if match:
        day = match.group(1).zfill(2)
        month = match.group(2).zfill(2)
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return f"{day}.{month}"
    
    return None


def is_ui_word(text):
    """Check if text is a UI word that should be filtered out."""
    text_lower = text.lower().strip()
    return any(ui_word in text_lower for ui_word in UI_WORDS)


def clean_name(name):
    """
    Clean an extracted name.
    
    - Strip whitespace
    - Remove trailing punctuation
    - Preserve accents
    - Return None if it's a UI word
    """
    name = name.strip()
    
    # Remove trailing punctuation
    name = re.sub(r'[,.\-:;!?]+$', '', name)
    
    # Remove leading/trailing whitespace again
    name = name.strip()
    
    # Filter out UI words
    if not name or is_ui_word(name) or len(name) < 2:
        return None
    
    return name


def extract_dates_and_names(text):
    """
    Parse OCR text to extract date headings and associated names.
    
    Returns a dict mapping date (DD.MM) to set of names.
    """
    results = defaultdict(set)
    lines = text.split('\n')
    
    current_date = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to parse as a date
        date = parse_date(line)
        if date:
            current_date = date
            continue
        
        # If we have a current date, try to extract names
        if current_date:
            # Clean the name
            cleaned = clean_name(line)
            if cleaned:
                results[current_date].add(cleaned)
    
    return results


def ocr_image(image):
    """
    Run Tesseract OCR on the image with French and English languages.
    
    Returns extracted text.
    """
    # Use French and English languages
    config = '--psm 6'  # Assume a single uniform block of text
    text = pytesseract.image_to_string(image, lang='fra+eng', config=config)
    return text


def process_image(owner, repo, branch, path, local_path=None):
    """
    Process a single image file.
    
    Returns dict mapping date to set of names.
    """
    try:
        # Download image
        image = download_image(owner, repo, branch, path, local_path)
        
        # Preprocess
        image = preprocess_image(image)
        
        # Run OCR
        text = ocr_image(image)
        
        # Extract dates and names
        results = extract_dates_and_names(text)
        
        if results:
            print(f"  Extracted {sum(len(names) for names in results.values())} names "
                  f"across {len(results)} dates")
        
        return results
    
    except Exception as e:
        print(f"  Error processing {path}: {e}")
        return {}


def merge_results(all_results):
    """
    Merge results from all images.
    
    Returns a single dict mapping date to set of names.
    """
    merged = defaultdict(set)
    
    for results in all_results:
        for date, names in results.items():
            merged[date].update(names)
    
    return merged


def write_csv(output_path, results):
    """
    Write results to CSV file.
    
    Format:
    - Header: JJ.MM,Prénoms
    - Date sorted chronologically (by month then day)
    - Names per row sorted alphabetically (case-insensitive)
    - Quote Prénoms field to preserve internal commas
    """
    # Sort dates chronologically (by month, then day)
    sorted_dates = sorted(results.keys(), key=lambda d: (d.split('.')[1], d.split('.')[0]))
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['JJ.MM', 'Prénoms'])
        
        for date in sorted_dates:
            names = results[date]
            # Sort names alphabetically (case-insensitive)
            sorted_names = sorted(names, key=lambda n: n.lower())
            # Join names with commas
            names_str = ', '.join(sorted_names)
            writer.writerow([date, names_str])
    
    print(f"\nWrote {len(sorted_dates)} dates to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get token from env if not provided
    token = args.token or os.environ.get('GITHUB_TOKEN')
    
    # Get list of image files
    image_files = get_image_files(args.owner, args.repo, args.branch, token, args.local_path)
    
    if not image_files:
        print("No image files found in repository")
        return 1
    
    # Process each image
    all_results = []
    for path in image_files:
        results = process_image(args.owner, args.repo, args.branch, path, args.local_path)
        all_results.append(results)
    
    # Merge all results
    merged_results = merge_results(all_results)
    
    if not merged_results:
        print("\nNo dates and names extracted")
        return 1
    
    # Write CSV
    write_csv(args.output, merged_results)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
