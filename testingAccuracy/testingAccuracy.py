from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import train_test_split
import itertools

from TFIDFmodel.AI import preprocess
from TFIDFmodel.main import reviews_docs

# Splitting the documents into training and test sets
train_docs, test_docs = train_test_split(reviews_docs, test_size=0.2, random_state=42)

# Preprocessing the documents
train_docs_processed = [preprocess(doc).split() for doc in train_docs]
test_docs_processed = [preprocess(doc).split() for doc in test_docs]

# train the Word2Vec model
model = Word2Vec(sentences=train_docs_processed, vector_size=100, window=5, min_count=2, workers=4)


# Function to get the average Word2Vec vectors for a document
def document_vector(doc):
    doc = [word for word in doc if word in model.wv.index_to_key]
    return np.mean(model.wv[doc], axis=0) if len(doc) > 0 else np.zeros(model.vector_size)


test_vectors = np.array([document_vector(doc) for doc in test_docs_processed])

categories = [
    ["morning", "afternoon", "evening"],
    ["group", "single", "family", "couple"],
    ["quiet", "crowded"],
    ["famous", "hidden", "popular", "normal"],
    ["all day", "half day", "couple of hours", "one hour"],
    ["historical", "natural", "cultural", "adventurous"],
    ["indoor", "outdoor"],
    ["relax", "energy"],
    ["independent", "guided"],
]

# generate the combinations
evaluation_docs = list(itertools.product(*categories))

# Preprocess docs
evaluation_docs_processed = [" ".join(doc).split() for doc in evaluation_docs]

# compute vectors
evaluation_vectors = np.array([document_vector(doc) for doc in evaluation_docs_processed])

# compute similarity scores
for eval_index, eval_vector in enumerate(evaluation_vectors):
    similarity_scores = cosine_similarity([eval_vector], test_vectors)

    # schoe scores
    print(f"Evaluation Document {eval_index}:")
    for i, score in enumerate(similarity_scores[0]):
        print(f"  Test Document {i} - Similarity Score: {score:.2f}")

    # evaluate
    mean_score = np.mean(similarity_scores)
    print(f"  Mean Similarity Score: {mean_score:.2f}")

    # in funcție de aplicatie, un scor >0.6 poate fi considerat bun
    good_matches = np.sum(similarity_scores > 0.6)
    print(f"  Number of good matches (>0.6): {good_matches} out of {len(similarity_scores[0])}")

overall_similarity_scores = cosine_similarity(evaluation_vectors, test_vectors)
overall_mean_score = np.mean(overall_similarity_scores)
print(f"Overall Mean Similarity Score for Evaluation Documents: {overall_mean_score:.2f}")

good_matches_overall = np.sum(overall_similarity_scores > 0.6)
total_comparisons = len(evaluation_vectors) * len(test_vectors)
print(f"Total Number of good matches (>0.6): {good_matches_overall} out of {total_comparisons}")
