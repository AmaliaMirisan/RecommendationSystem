import re

from unidecode import unidecode


def normalize_name(name):
    # Normalize the name by removing diacritics
    name = unidecode(name)
    # Remove specific characters and replace spaces with underscores
    name = re.sub(r'[ ,\-./:*&()!]', '_', name.lower())
    name = re.sub(r'__+', '_', name)  # Replace multiple underscores with a single underscore
    name = name.replace("'", "_")
    return name.strip('_')  # Remove leading and trailing underscores if any

# Function to find the link for a given attraction name
def find_link(attractions, processed_name):
    for attraction in attractions:
        link = attraction['attraction'].lower()
        if processed_name in link:
            return link
    return "No link found"