import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed, parallel_backend
import warnings
import joblib
import pickle

warnings.filterwarnings("ignore", category=UserWarning)

# Încărcarea datelor preprocesate
with open('C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\AdvancedTuningTF-IDF\\processed_reviews.pkl',
          'rb') as file_reviews:
    reviews = pickle.load(file_reviews)

with open('C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\AdvancedTuningTF-IDF\\user_input_combinations.pkl',
          'rb') as file_combinations:
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

# Funcția de scor pentru RandomizedSearchCV
def custom_search(estimator, X):
    tfidf_matrix = estimator.transform(X)
    return similarity_score(tfidf_matrix, combinations, estimator)
param_dist = {
        'max_features': [25, 30, 35],
        'ngram_range': [(1, 1), (1, 2), (1, 3)],
        'max_df': [0.6, 0.7, 0.8],
        'min_df': [2, 3, 5],
        'norm': [None, 'l1', 'l2'],
        'stop_words': ['english']
    }
try:

    print("-------------------------Starting randomized search-------------------------")
    # Randomized search pentru găsirea celor mai buni parametri
    random_search = RandomizedSearchCV(
        TfidfVectorizer(use_idf=True),
        param_distributions=param_dist,
        n_iter=100,  # Numărul de combinații aleatorii de parametri
        cv=3,
        scoring=lambda estimator, X: custom_search(estimator, X),
        verbose=10,
    )

    random_search.fit(reviews)
    best_params = random_search.best_params_
    print("Best parameters found: ", best_params)

    # Salvarea modelului folosind joblib
    joblib_path = 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\AdvancedTuningTF-IDF\\best_tfidf_model.joblib'
    joblib.dump(random_search, joblib_path)
except KeyboardInterrupt:
    # Salvarea modelului și a rezultatelor parțiale în caz de întrerupere
    joblib.dump(random_search, 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\AdvancedTuningTF-IDF\\interrupted_tfidf_model.joblib')
    print("Process interrupted. Model partially saved.")
except Exception as e:
    joblib.dump(random_search,
                'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\AdvancedTuningTF-IDF\\interrupted_tfidf_model.joblib')
    print("Process interrupted. Model partially saved.")
    print(f"An error occurred: {e}")

'''[CV 1/3; 13/100] END max_df=0.7, max_features=35, min_df=3, ngram_range=(1, 2), norm=l2, stop_words=english;, score=0.187 total time= 2.1min
[CV 2/3; 13/100] START max_df=0.7, max_features=35, min_df=3, ngram_range=(1, 2), norm=l2, stop_words=english

[CV 1/3; 30/100] END max_df=0.7, max_features=35, min_df=3, ngram_range=(1, 2), norm=l1, stop_words=english;, score=0.187 total time= 4.3min
[CV 2/3; 30/100] START max_df=0.7, max_features=35, min_df=3, ngram_range=(1, 2), norm=l1, stop_words=english

[CV 1/3; 46/100] END max_df=0.7, max_features=35, min_df=5, ngram_range=(1, 2), norm=None, stop_words=english;, score=0.187 total time= 4.4min
[CV 2/3; 46/100] START max_df=0.7, max_features=35, min_df=5, ngram_range=(1, 2), norm=None, stop_words=english

[CV 1/3; 48/100] END max_df=0.7, max_features=35, min_df=2, ngram_range=(1, 3), norm=None, stop_words=english;, score=0.187 total time= 4.3min
[CV 2/3; 48/100] START max_df=0.7, max_features=35, min_df=2, ngram_range=(1, 3), norm=None, stop_words=english
'''

'''--------SECOND:
[CV 1/3; 46/100] END max_df=0.7, max_features=25, min_df=3, ngram_range=(1, 1), norm=None, stop_words=english;, score=0.212 total time= 1.9min
[CV 1/3; 49/100] END max_df=0.7, max_features=25, min_df=2, ngram_range=(1, 1), norm=l1, stop_words=english;, score=0.212 total time= 2.0min
[CV 1/3; 51/100] END max_df=0.7, max_features=25, min_df=2, ngram_range=(1, 1), norm=None, stop_words=english;, score=0.212 total time= 2.0min
[CV 1/3; 55/100] END max_df=0.7, max_features=25, min_df=5, ngram_range=(1, 1), norm=l2, stop_words=english;, score=0.212 total time= 2.0min
[CV 1/3; 92/100] END max_df=0.7, max_features=25, min_df=3, ngram_range=(1, 1), norm=l2, stop_words=english;, score=0.212 total time= 2.1min
[CV 1/3; 95/100] END max_df=0.7, max_features=25, min_df=5, ngram_range=(1, 1), norm=l1, stop_words=english;, score=0.212 total time= 2.1min
'''