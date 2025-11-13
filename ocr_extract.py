#!/usr/bin/env python3
"""
OCR extraction script for French name days (fêtes) from screenshot images.

This script:
1. Uses GitHub Git Trees API to list all image files in the repository
2. Downloads and processes images with OCR (pytesseract)
3. Extracts dates and associated first names
4. Produces a CSV file with columns: JJ.MM and Prénoms
"""

import argparse
import csv
import io
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import requests
from PIL import Image
import pytesseract


# French month name to number mapping
FRENCH_MONTHS = {
    'janvier': '01', 'jan': '01', 'janv': '01',
    'février': '02', 'fevrier': '02', 'fév': '02', 'fev': '02', 'févr': '02', 'fevr': '02',
    'mars': '03', 'mar': '03',
    'avril': '04', 'avr': '04',
    'mai': '05',
    'juin': '06', 'jun': '06',
    'juillet': '07', 'juil': '07', 'juill': '07',
    'août': '08', 'aout': '08', 'aoû': '08', 'aou': '08',
    'septembre': '09', 'sept': '09', 'sep': '09',
    'octobre': '10', 'oct': '10',
    'novembre': '11', 'nov': '11',
    'décembre': '12', 'decembre': '12', 'déc': '12', 'dec': '12'
}

# Words to filter out (UI elements, not names)
FILTER_WORDS = {
    'mes fêtes', 'mes fetes', 'favoris', 'ajouter', 'fêtes à souhaiter',
    'fetes a souhaiter', 'fêtes', 'fetes', 'souhaiter', 'lite',
    'rechercher', 'partager', 'modifier', 'supprimer', 'annuler',
    'valider', 'retour', 'suivant', 'précédent', 'precedent'
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract name days from repository screenshots using OCR'
    )
    parser.add_argument('--owner', required=True, help='Repository owner')
    parser.add_argument('--repo', required=True, help='Repository name')
    parser.add_argument('--branch', default='main', help='Branch name (default: main)')
    parser.add_argument('--output', default='fetes_extracted.csv',
                        help='Output CSV file (default: fetes_extracted.csv)')
    parser.add_argument('--token', default=None,
                        help='GitHub token (default: from GITHUB_TOKEN env var)')
    return parser.parse_args()


def get_image_files(owner: str, repo: str, branch: str, token: str = None) -> List[str]:
    """
    Use GitHub Git Trees API to list all image files in the repository.
    
    Returns list of image file paths.
    """
    url = f'https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1'
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    print(f"Fetching file tree from {url}...", file=sys.stderr)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    tree_data = response.json()
    image_files = []
    
    for item in tree_data.get('tree', []):
        if item['type'] == 'blob':
            path = item['path']
            # Match common image extensions
            if any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
                image_files.append(path)
    
    print(f"Found {len(image_files)} image files", file=sys.stderr)
    return image_files


def download_image(owner: str, repo: str, branch: str, path: str) -> Image.Image:
    """
    Download image from raw.githubusercontent.com.
    
    Returns PIL Image object.
    """
    url = f'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}'
    print(f"Downloading {path}...", file=sys.stderr)
    
    response = requests.get(url)
    response.raise_for_status()
    
    image = Image.open(io.BytesIO(response.content))
    return image


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy.
    
    - Convert to grayscale
    - Resize if needed
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Optional: resize if image is too small
    min_dimension = 800
    if image.width < min_dimension or image.height < min_dimension:
        scale = min_dimension / min(image.width, image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image


def normalize_month(month_str: str) -> str:
    """
    Normalize French month name to two-digit month number.
    
    Returns '01' through '12' or None if not found.
    """
    month_lower = month_str.lower().strip()
    return FRENCH_MONTHS.get(month_lower)


def parse_date_heading(line: str) -> Tuple[str, str]:
    """
    Try to parse a date heading from a line.
    
    Supported formats:
    - "26 Juillet" -> "26.07"
    - "26.07" -> "26.07"
    - "26 Juil" -> "26.07"
    - "26/07" -> "26.07"
    
    Returns (day, month) as two-digit strings or (None, None) if not a date.
    """
    line = line.strip()
    
    # Pattern 1: DD.MM or DD/MM
    match = re.match(r'^(\d{1,2})[./](\d{1,2})$', line)
    if match:
        day = match.group(1).zfill(2)
        month = match.group(2).zfill(2)
        return day, month
    
    # Pattern 2: DD Month_Name
    match = re.match(r'^(\d{1,2})\s+([a-zàâäéèêëïîôùûüçœæ]+)', line, re.IGNORECASE)
    if match:
        day = match.group(1).zfill(2)
        month_name = match.group(2)
        month = normalize_month(month_name)
        if month:
            return day, month
    
    return None, None


def clean_name(name: str) -> str:
    """
    Clean an extracted name.
    
    - Strip whitespace
    - Remove trailing punctuation
    - Preserve accents
    """
    name = name.strip()
    # Remove trailing punctuation (but preserve hyphens in names like Jean-Paul)
    name = re.sub(r'[.,;:!?]+$', '', name)
    return name


def is_valid_name(name: str) -> bool:
    """
    Check if a string looks like a valid first name.
    
    - Not empty
    - Not a UI word
    - Mostly alphabetic (allowing hyphens and accents)
    - At least 2 characters
    """
    if not name or len(name) < 2:
        return False
    
    # Check against filter words
    name_lower = name.lower()
    if name_lower in FILTER_WORDS:
        return False
    
    # Check if it's mostly letters (allow hyphens, spaces for compound names)
    if not re.match(r'^[a-zàâäéèêëïîôùûüçœæ\s\-]+$', name, re.IGNORECASE):
        return False
    
    return True


def extract_from_ocr_text(text: str) -> Dict[str, Set[str]]:
    """
    Parse OCR text to extract dates and associated names.
    
    Returns dict mapping date keys (DD.MM) to sets of names.
    """
    results = defaultdict(set)
    lines = text.split('\n')
    
    current_date = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to parse as date heading
        day, month = parse_date_heading(line)
        if day and month:
            current_date = f"{day}.{month}"
            continue
        
        # If we have a current date, treat this line as a potential name
        if current_date:
            # Split on common separators
            potential_names = re.split(r'[,;]', line)
            for name in potential_names:
                name = clean_name(name)
                if is_valid_name(name):
                    results[current_date].add(name)
    
    return results


def process_image(image: Image.Image) -> Dict[str, Set[str]]:
    """
    Process a single image with OCR and extract dates/names.
    
    Returns dict mapping date keys to sets of names.
    """
    # Preprocess
    image = preprocess_image(image)
    
    # Run OCR with French and English
    try:
        text = pytesseract.image_to_string(image, lang='fra+eng')
        print(f"OCR extracted {len(text)} characters", file=sys.stderr)
    except Exception as e:
        print(f"OCR failed: {e}", file=sys.stderr)
        return {}
    
    # Parse the text
    return extract_from_ocr_text(text)


def merge_results(all_results: List[Dict[str, Set[str]]]) -> Dict[str, Set[str]]:
    """
    Merge results from multiple images.
    
    Combines all names for each date, deduplicating.
    """
    merged = defaultdict(set)
    for result in all_results:
        for date, names in result.items():
            merged[date].update(names)
    return merged


def sort_date_key(date_str: str) -> Tuple[int, int]:
    """
    Convert date string DD.MM to sortable tuple (month, day).
    
    This sorts chronologically by month, then by day.
    """
    day, month = date_str.split('.')
    return (int(month), int(day))


def write_csv(results: Dict[str, Set[str]], output_path: str):
    """
    Write results to CSV file.
    
    Format:
    - Header: JJ.MM,Prénoms
    - Dates sorted chronologically (by month, then day)
    - Names per row sorted alphabetically (case-insensitive)
    - Names joined by commas
    - Prénoms field quoted
    """
    # Sort dates chronologically
    sorted_dates = sorted(results.keys(), key=sort_date_key)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['JJ.MM', 'Prénoms'])
        
        for date in sorted_dates:
            names = results[date]
            # Sort names alphabetically (case-insensitive)
            sorted_names = sorted(names, key=lambda x: x.lower())
            # Join with comma
            names_str = ', '.join(sorted_names)
            writer.writerow([date, names_str])
    
    print(f"Wrote {len(sorted_dates)} dates to {output_path}", file=sys.stderr)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get token from argument or environment
    token = args.token or os.environ.get('GITHUB_TOKEN')
    
    # Get list of image files
    image_files = get_image_files(args.owner, args.repo, args.branch, token)
    
    if not image_files:
        print("No image files found in repository", file=sys.stderr)
        sys.exit(1)
    
    # Process each image
    all_results = []
    for path in image_files:
        try:
            image = download_image(args.owner, args.repo, args.branch, path)
            result = process_image(image)
            all_results.append(result)
            print(f"Extracted {sum(len(names) for names in result.values())} names from {path}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)
            continue
    
    # Merge all results
    merged = merge_results(all_results)
    
    if not merged:
        print("No data extracted from images", file=sys.stderr)
        sys.exit(1)
    
    # Write CSV
    write_csv(merged, args.output)
    
    print(f"Success! Extracted {len(merged)} dates with {sum(len(names) for names in merged.values())} total names", file=sys.stderr)


if __name__ == '__main__':
    main()
