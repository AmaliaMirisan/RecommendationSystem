import json
import pickle
from TFIDFmodel.LoadData import load_data

# Load the original data that contains city and province information
details_files = [
    'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_details_batch1.json',
    'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_details_batch2.json'
]
reviews_files = [
    'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_reviews_batch1.json',
    'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_reviews_batch2.json'
]
all_attractions = load_data(details_files, reviews_files)

# Extract unique city-province pairs from the original data
unique_locations = set()
negative_rating_locations = set()
negative_rating_ids = set()  # To store IDs of attractions with rating -1

for attraction in all_attractions:
    if 'city' in attraction and 'province' in attraction:
        if attraction['rating'] > 1:
            unique_locations.add((attraction['city'], attraction['province']))
        elif attraction['rating'] == -1:
            negative_rating_locations.add((attraction['city'], attraction['province'], attraction['attraction_id']))
            negative_rating_ids.add(attraction['attraction_id'])

# Create a list of dictionaries with unique IDs for each city-province pair
location_list = [{"id": idx + 1, "city": loc[0], "province": loc[1]} for idx, loc in enumerate(unique_locations)]

# Save the list of city-province pairs to a JSON file in a valid directory
output_path = 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\prepareFrontEnd\\unique_cities.json'
with open(output_path, 'w') as f:
    json.dump(location_list, f, indent=4)

print(f"Saved {len(unique_locations)} unique city-province pairs to {output_path}")

# Create a list of dictionaries with unique IDs for each city-province pair with rating -1
negative_rating_list = [{"id": loc[2], "city": loc[0], "province": loc[1]} for loc in negative_rating_locations]

# Save the list of negative rating city-province pairs to a JSON file in a valid directory
negative_output_path = 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\prepareFrontEnd\\negative_rating_cities.json'
with open(negative_output_path, 'w') as f:
    json.dump(negative_rating_list, f, indent=4)

print(f"Saved {len(negative_rating_locations)} unique city-province pairs with rating -1 to {negative_output_path}")

# Save the list of IDs for attractions with rating -1 to a JSON file in a valid directory
negative_ids_output_path = 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\prepareFrontEnd\\negative_rating_ids.json'
with open(negative_ids_output_path, 'w') as f:
    json.dump(list(negative_rating_ids), f, indent=4)

print(f"Saved {len(negative_rating_ids)} IDs of attractions with rating -1 to {negative_ids_output_path}")
