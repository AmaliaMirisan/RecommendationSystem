# Load JSON data for attractions
import json


def load_json_data_for_links(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def load_data(details_files, reviews_files):
    #     Mapping of attraction names to their details,
    #     including a list of reviews and a set of IDs
    name_to_details = {}
    id_to_name = {}  # Mapping of IDs to attraction names

    # Load attraction details
    for file in details_files:
        with open(file, 'r') as f:
            for entry in json.load(f):
                name = entry['name'].strip().lower()
                if name not in name_to_details:
                    name_to_details[name] = entry
                    name_to_details[name]['reviews'] = []
                    name_to_details[name]['ids'] = set()

                name_to_details[name]['ids'].add(entry['attraction_id'])
                id_to_name[entry['attraction_id']] = name

                # Ensure we always take the lowest price
                if 'price' in entry:
                    if 'price' in name_to_details[name]:
                        name_to_details[name]['price'] = min(name_to_details[name]['price'], entry['price'])
                    else:
                        name_to_details[name]['price'] = entry['price']

    # Load reviews and assign them to the correct attraction by name using the ID to name mapping
    for file in reviews_files:
        with open(file, 'r') as f:
            for review in json.load(f):
                if review['attraction_id'] in id_to_name:
                    name = id_to_name[review['attraction_id']]
                    name_to_details[name]['reviews'].append(review['review'])

    return list(name_to_details.values())


def filter_attractions(attractions, city, budget):
    # Initialize two lists: one for attractions with reviews and another for those without
    filtered_with_reviews = []
    filtered_no_reviews = []

    # Iterate over each attraction to check if it matches the city and budget criteria
    for attr in attractions:
        if attr['city'].lower() == city.lower() and attr['price'] > -1 and attr['price'] <= budget:
            # Check if the attraction has reviews
            if attr['reviews'] and attr['rating'] > 1:
                filtered_with_reviews.append(attr)
            else:
                filtered_no_reviews.append(attr)

    # Return both lists
    return filtered_with_reviews, filtered_no_reviews

def filter_attractions_cluster(attractions, city, budget):
    # Initialize two lists: one for attractions with reviews and another for those without
    filtered_with_reviews = []
    filtered_no_reviews = []

    for attr in attractions:
        if attr['reviews'] and attr['rating'] > 1:
            filtered_with_reviews.append(attr)
        else:
             filtered_no_reviews.append(attr)

    # Return both lists
    return filtered_with_reviews, filtered_no_reviews