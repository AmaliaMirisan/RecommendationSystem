import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pickle

# Load the preprocessed data again
with open('C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\AdvancedTuningTF-IDF\\processed_reviews.pkl', 'rb') as file_reviews:
    reviews = pickle.load(file_reviews)

with open('C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\AdvancedTuningTF-IDF\\user_input_combinations.pkl', 'rb') as file_combinations:
    combinations = pickle.load(file_combinations)

def similarity_score(tfidf_matrix, preprocessed_combinations, estimator):
    scores = []
    for combo in preprocessed_combinations:
        user_vector = estimator.transform([combo])
        cosine_sim = cosine_similarity(tfidf_matrix, user_vector)
        mean_cosine_sim = np.mean(cosine_sim)
        scores.append(mean_cosine_sim)
    overall_mean_score = np.mean(scores)
    return overall_mean_score

# Use the best parameters found in your previous run
best_params = {
    'stop_words': 'english',
    'norm': 'l2',
    'ngram_range': (1, 1),
    'min_df': 3,
    'max_features': 25,
    'max_df': 0.7
}

# Initialize the TfidfVectorizer with the best parameters
vectorizer = TfidfVectorizer(
    stop_words=best_params['stop_words'],
    norm=best_params['norm'],
    ngram_range=best_params['ngram_range'],
    min_df=best_params['min_df'],
    max_features=best_params['max_features'],
    max_df=best_params['max_df']
)

# Fit the TfidfVectorizer with the reviews
tfidf_matrix = vectorizer.fit_transform(reviews)

# Save the fitted TfidfVectorizer
joblib_path = 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\AdvancedTuningTF-IDF\\best_tfidf_model.joblib'
joblib.dump(vectorizer, joblib_path)

print("Model successfully saved with the best parameters.")
