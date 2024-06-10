import joblib
from sklearn.metrics.pairwise import cosine_similarity

from LoadData import load_data, load_json_data_for_links, filter_attractions, filter_attractions_cluster
from AI import compare_user_input_tfidf, create_tfidf_vectorizer, preprocess_stop_words, preprocess_user_input, \
    preprocess, \
    compare_user_input_euclidean, get_embeddings, combined_similarity_score, compare_user_input_embeddings
from Print import find_link, normalize_name

# Load English tokenizer, tagger, parser, NER, and word vectors
attractions_links = load_json_data_for_links('C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_cat.json')  # Update the path as necessary

# Load data
details_files = ['C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_details_batch1.json', 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_details_batch2.json']
reviews_files = ['C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_reviews_batch1.json', 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_reviews_batch2.json']
all_attractions = load_data(details_files, reviews_files)

# User input
user_city = "toronto"
user_budget = 200.0
#user_input = "I love beach holidays and sunny destinations."
# Example user input as a list of responses
user_responses = ["tour", "guid", "nature", "adventure", "history", "funny"]

# Combine into a single string
#combined_responses = ", ".join(user_responses)
user_input_processed = preprocess_user_input(user_responses)
#user_input_processed = "i love hiking and i also love to go in trips with my family. I want a full day vacation with a lot of culture and sport"

# Filter attractions based on user input
#filtered_attractions, filtered_no_reviews = filter_attractions(all_attractions, user_city, user_budget)
filtered_attractions, filtered_no_reviews = filter_attractions(all_attractions, user_city, user_budget)


# Assuming filtered_attractions is the list of attractions that passed the city and budget filters and have reviews
reviews_docs = [attraction['reviews'] for attraction in filtered_attractions if 'reviews' in attraction]

# Flatten the list of reviews per attraction into a single string per attraction
reviews_docs = [" ".join(reviews) for reviews in reviews_docs]

# Load the previously saved TF-IDF vectorizer model
vectorizer_path = 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\AdvancedTuningTF-IDF\\best_tfidf_model.joblib'
vectorizer = joblib.load(vectorizer_path)
# Transform the reviews using the loaded TF-IDF vectorizer
tfidf_matrix = vectorizer.transform(reviews_docs)

# Create TF-IDF vectorizer and vectorize the documents
#vectorizer, tfidf_matrix = create_tfidf_vectorizer(reviews_docs)


# Compare user input
#similarity_scores = compare_user_input(vectorizer, user_input_processed, tfidf_matrix)

tfidf_scores = compare_user_input_tfidf(vectorizer, user_input_processed, tfidf_matrix)

# generate embeddings for the same docs and user_input
embedding_scores = compare_user_input_embeddings(reviews_docs, user_input_processed)

# combine scores and find best recommendations
combined_scores = combined_similarity_score(tfidf_scores, embedding_scores)

top_indices = combined_scores.argsort()[0][::-1]  # scores indices in an descending order

print("Top recommended attractions based on your input:")
for idx in top_indices[:5]:  # Adjust the number of top attractions as needed
    attraction = filtered_attractions[idx]
    processed_name = normalize_name(attraction['name'])
    link = find_link(attractions_links, processed_name)
    print(f"{attraction['name']} - Score: {combined_scores[0][idx]:.2f}")
    print(f"  Location: {attraction['city']}")
    print(f"  Price: {attraction['price'] if 'price' in attraction else 'Not available'}")
    print(f"  Link: {link if link != 'No link found' else 'Link not available'}")

