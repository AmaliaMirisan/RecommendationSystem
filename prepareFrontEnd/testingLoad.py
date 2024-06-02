import json
import pickle

# Load the original data that contains city and province information
details_files = [
    'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_details_batch1.json',
    'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_details_batch2.json'
]
reviews_files = [
    'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_reviews_batch1.json',
    'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_reviews_batch2.json'
]

def load_data(details_files, reviews_files):
    name_to_details = {}
    id_to_name = {}

    # Load attraction details
    for file in details_files:
        with open(file, 'r', encoding='utf-8') as f:
            for entry in json.load(f):
                name = entry['name'].strip().lower()
                if name not in name_to_details:
                    name_to_details[name] = entry
                    name_to_details[name]['reviews'] = []
                    name_to_details[name]['ids'] = set()

                name_to_details[name]['ids'].add(entry['attraction_id'])
                id_to_name[entry['attraction_id']] = name

                if 'price' in entry:
                    if 'price' in name_to_details[name]:
                        name_to_details[name]['price'] = min(name_to_details[name]['price'], entry['price'])
                    else:
                        name_to_details[name]['price'] = entry['price']

    # Load reviews
    for file in reviews_files:
        with open(file, 'r', encoding='utf-8') as f:
            for review in json.load(f):
                if review['attraction_id'] in id_to_name:
                    name = id_to_name[review['attraction_id']]
                    name_to_details[name]['reviews'].append(review['review'])

    return list(name_to_details.values())

all_attractions = load_data(details_files, reviews_files)

# Verify the completeness of the reviews by checking their lengths
for i, attraction in enumerate(all_attractions[:5]):
    if 'reviews' in attraction:
        print(f"Attraction {i + 1}: {attraction['name']}")
        print(f"Number of reviews: {len(attraction['reviews'])}")
        for review in attraction['reviews'][:3]:  # Check the first 3 reviews for brevity
            print(f"Review length: {len(review)} characters")
        print("\n")
