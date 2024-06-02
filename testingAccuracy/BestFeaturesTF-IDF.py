from LoadData import load_data, load_json_data_for_links, filter_attractions_cluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings

'''
BEST
param_grid = {
    'tfidf__max_features': 5000,
    'tfidf__ngram_range': (1, 3),
    'tfidf__max_df': 0.5,
    'tfidf__min_df': 5,
    'tfidf__stop_words': None,
    'tfidf__use_idf': True,
    'tfidf__norm': l1
}'''
warnings.filterwarnings("ignore")
# Load and preprocess data as you've outlined
attractions_links = load_json_data_for_links('../attractions_cat.json')
details_files = ['attractions_details_batch1.json', 'attractions_details_batch2.json']
reviews_files = ['attractions_reviews_batch1.json', 'attractions_reviews_batch2.json']
all_attractions = load_data(details_files, reviews_files)


filtered_attractions, filtered_no_reviews = filter_attractions_cluster(all_attractions)
reviews_docs = [attraction['reviews'] for attraction in filtered_attractions if 'reviews' in attraction]
reviews_docs = [" ".join(reviews) for reviews in reviews_docs]

# Define the TF-IDF vectorizer and KMeans clustering within a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('kmeans', KMeans(n_clusters=5, random_state=42))  # Choose an appropriate number of clusters
])

# Define the parameter grid

param_grid = {
    'tfidf__max_features': [1000, 2000, 3000, 4000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__max_df': [0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0],
    'tfidf__min_df': [1, 2, 3, 5, 7, 10],
    'tfidf__stop_words': [None, 'english'],
    'tfidf__use_idf': [True, False],
    'tfidf__norm': ['l1', 'l2']
}
# Setup grid search; use silhouette score as a custom scoring function
from sklearn.base import clone

# Define a custom scorer that properly uses the transformed data for scoring
def silhouette_scorer(estimator, X):
    # Clone the estimator to ensure the original pipeline remains unmodified
    cloned_estimator = clone(estimator)
    # Fit the cloned estimator to the data
    cloned_estimator.fit(X)
    # Get cluster labels
    labels = cloned_estimator.named_steps['kmeans'].labels_
    # Get the transformed data (TF-IDF vectors)
    tfidf_vectors = cloned_estimator.named_steps['tfidf'].transform(X)
    # Calculate silhouette score using the actual feature set and labels
    score = silhouette_score(tfidf_vectors, labels)
    return score

# Now setup and run your grid search as before
grid_search = GridSearchCV(pipeline, param_grid, scoring=silhouette_scorer, cv=3, verbose=10)
grid_search.fit(reviews_docs)
print("Best parameters found:")
print(grid_search.best_params_)
print("Best silhouette score:", grid_search.best_score_)

from joblib import dump

# După antrenarea modelului și vectorizatorului TF-IDF, salvează-le pe disc
# Salvează modelul KMeans
dump(grid_search.best_estimator_.named_steps['kmeans'], 'kmeans_model.joblib')

# Salvează vectorizatorul TF-IDF
dump(grid_search.best_estimator_.named_steps['tfidf'], 'tfidf_vectorizer.joblib')
