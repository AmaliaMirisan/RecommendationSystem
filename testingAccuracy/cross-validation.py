# Configurarea TF-IDF Vectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from torch import cosine_similarity

from main import reviews_docs

tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=1,
    max_df=0.9,
    ngram_range=(1, 1),
    stop_words='english'
)

# Transformarea documentelor în TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_docs)

# Definirea numărului de fold-uri pentru cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Funcție pentru a calcula scorurile de similaritate și performanța generală
def evaluate_model(train_index, test_index):
    X_train, X_test = tfidf_matrix[train_index], tfidf_matrix[test_index]
    similarity_scores = cosine_similarity(X_test, X_train)

    # Afișarea scorurilor de similaritate pentru fiecare fold
    mean_scores = np.mean(similarity_scores, axis=1)
    print(f"Mean Similarity Scores for this fold: {mean_scores}")

    return mean_scores


# Listă pentru a stoca scorurile medii pentru fiecare fold
all_mean_scores = []

# Aplicarea cross-validation
for train_index, test_index in kf.split(tfidf_matrix):
    mean_scores = evaluate_model(train_index, test_index)
    all_mean_scores.extend(mean_scores)

# Calcularea și afișarea scorului mediu final
overall_mean_score = np.mean(all_mean_scores)
print(f"Overall Mean Similarity Score: {overall_mean_score:.2f}")

# Evaluarea numărului de "good matches" (>0.6) pentru toate fold-urile
good_matches = np.sum(np.array(all_mean_scores) > 0.6)
print(f"Number of good matches (>0.6): {good_matches} out of {len(all_mean_scores)}")