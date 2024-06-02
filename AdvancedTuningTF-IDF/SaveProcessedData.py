import itertools

# Define the terms
terms = [
    {"morning", "afternoon", "evening"},
    {"spring", "summer", "autumn", "winter"},
    {"all day", "half-day", "hours"},
    {"sunny", "rainy", "snowy"},
    {"group", "single", "family", "couple"},
    {"indoor", "outdoor"},
    {"independent", "guide"},
    {"quiet", "crowd"},
    {"famous", "hidden", "popular", "normal"},
    {"history", "natural", "cultural", "adventure", "food", "wine"},
    {"relax", "energy"}
]

# Function to generate and save all unique combinations
def generate_and_save_combinations(terms, filename="user_input_combinations.pkl"):
    all_combinations = set()

    # Calculate total number of combinations
    total_combinations = 1
    for term_set in terms:
        total_combinations *= len(term_set)

    with tqdm(total=total_combinations, desc="Processing combinations") as pbar:
        for combination in itertools.product(*terms):
            # Convert combination to a sorted tuple to ensure uniqueness
            sorted_combination = tuple(sorted(combination))
            # Join the sorted tuple into a single string
            combination_string = " ".join(sorted_combination)
            # Add to the set of unique combinations
            all_combinations.add(combination_string)
            pbar.update(1)

    # Save the unique combinations to a pickle file
    with open(filename, "wb") as f:
        pickle.dump(all_combinations, f)

    print(f"Saved {len(all_combinations)} unique combinations to {filename}")

# Function to load and check for duplicates
def check_for_duplicates(filename="user_input_combinations.pkl"):
    with open(filename, "rb") as f:
        all_combinations = pickle.load(f)

    # Use a set to track seen combinations
    seen_combinations = set()

    for combination in all_combinations:
        if combination in seen_combinations:
            raise ValueError(f"Duplicate combination found: {combination}")
        seen_combinations.add(combination)

    print("No duplicates found.")


# Generate and save all unique combinations
generate_and_save_combinations(terms)


# Check for duplicates in the pickle file
check_for_duplicates()

import pickle
import spacy
from nltk.stem import WordNetLemmatizer
from TFIDFmodel.LoadData import load_json_data_for_links, load_data
from tqdm import tqdm

# Încarcă modelul spaCy și inițializează lematizatorul
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

# Funcția de preprocesare a textului
def preprocess(text):
    doc = nlp(text.lower())
    lemmatized = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(lemmatized)

# Încarcă datele de recenzii și filtrează
attractions_links = load_json_data_for_links('../attractions_cat.json')
details_files = ['../attractions_details_batch1.json', '../attractions_details_batch2.json']
reviews_files = ['../attractions_reviews_batch1.json', '../attractions_reviews_batch2.json']
all_attractions = load_data(details_files, reviews_files)

# Filtrarea atracțiilor în funcție de recenzii și rating
def filter_attr(attractions):
    filtered_with_reviews = []
    filtered_no_reviews = []
    for attr in attractions:
        if attr['reviews'] and attr['rating'] > 1:
            filtered_with_reviews.append(attr)
        else:
            filtered_no_reviews.append(attr)
    return filtered_with_reviews, filtered_no_reviews

filtered_attractions, filtered_no_reviews = filter_attr(all_attractions)
reviews_docs = [" ".join(attraction['reviews']) for attraction in filtered_attractions if 'reviews' in attraction]

# Preprocesarea documentelor de recenzii cu progress bar
processed_docs = []
for doc in tqdm(reviews_docs, desc="Preprocessing reviews"):
    processed_docs.append(preprocess(doc))

# Salvarea datelor preprocesate într-un fișier
with open('processed_reviews.pkl', 'wb') as f:
    pickle.dump(processed_docs, f)

print("Reviews preprocessed and saved.")
