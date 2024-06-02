import itertools
from tqdm import tqdm
import pickle

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

