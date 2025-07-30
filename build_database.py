import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin
import json
from PIL import Image
import io

# --- 1. CONFIGURATION ---
WEBSITE_URL = 'https://www.mmmut.ac.in/FacultyList?ab=11'
FACULTY_CONTAINER_SELECTOR = 'div.row.m-0[style*="border"]'
NAME_SELECTOR = 'h5[style*="font-weight: bolder"]'
DESIGNATION_SELECTOR = 'span[id*="Label4"]' # Selector for the designation
IMAGE_SELECTOR = 'img.img_th'
DB_PATH = "faculty_db"

# --- END OF CONFIGURATION ---

def build_database():
    """
    Scrapes the website, downloads images, and creates a JSON file with faculty details.
    """
    print(f"Scraping faculty data from: {WEBSITE_URL}")
    
    try:
        page = requests.get(WEBSITE_URL, headers={'User-Agent': 'Mozilla/5.0'})
        page.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not access the URL: {e}")
        return

    soup = BeautifulSoup(page.content, 'html.parser')
    faculty_list = soup.select(FACULTY_CONTAINER_SELECTOR)
    
    if not faculty_list:
        print("Error: Could not find any faculty members. Check selectors.")
        return
        
    print(f"Found {len(faculty_list)} faculty members.")

    faculty_data = {} # Dictionary to hold all our data

    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)

    for faculty_card in faculty_list:
        try:
            name_element = faculty_card.select_one(NAME_SELECTOR)
            designation_element = faculty_card.select_one(DESIGNATION_SELECTOR)
            img_element = faculty_card.select_one(IMAGE_SELECTOR)

            if not all([name_element, designation_element, img_element]):
                print("Warning: Skipping a card, missing required elements.")
                continue

            name = name_element.get_text(strip=True)
            designation = designation_element.get_text(strip=True).title() # Capitalize words
            department = "Civil Engineering"
            safe_name = re.sub(r'[^\w\s-]', '', name).strip()

            if not safe_name:
                print(f"Warning: Skipping entry for '{name}' due to invalid name.")
                continue

            # Add data to our dictionary
            faculty_data[safe_name] = {
                "full_name": name,
                "designation": designation,
                "department": department
            }

            img_url = img_element['src']
            if not img_url.startswith(('http:', 'https:')):
                img_url = urljoin(WEBSITE_URL, img_url)

            img_response = requests.get(img_url)
            image_stream = io.BytesIO(img_response.content)
            img = Image.open(image_stream).convert('RGB')
            
            # Save the image with the safe_name, which we use as an ID
            img_path = os.path.join(DB_PATH, f"{safe_name}.jpg")
            img.save(img_path, "JPEG")
            
            print(f"Saved photo and data for: {name}")

        except Exception as e:
            print(f"An error occurred while processing a card for '{name}': {e}")

    # Save the collected data to a JSON file
    with open('faculty_data.json', 'w') as f:
        json.dump(faculty_data, f, indent=4)
    print("\nSuccessfully saved faculty details to faculty_data.json")
    print(f"Database build complete! Images are in the '{DB_PATH}' folder.")

if __name__ == '__main__':
    build_database()
