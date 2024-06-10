import re
from nltk.corpus import wordnet

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    # Lowercase, lemmatize, remove stop words, punctuation, and non-alphabetic characters
    doc = nlp(text.lower())
    lemmatized = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(lemmatized)


def preprocess_stop_words(stop_words, nlp, lemmatizer):
    processed_stop_words = set()
    for word in stop_words:
        doc = nlp(word.lower())  # Ensure lowercase for consistency
        for token in doc:
            if not token.is_stop and not token.is_punct:
                processed_stop_words.add(lemmatizer.lemmatize(token.lemma_))
    return processed_stop_words


custom_stop_words = set(stopwords.words('english')).union({"additional", "words", "specific", "context"})
processed_stop_words = preprocess_stop_words(custom_stop_words, nlp, lemmatizer)


def preprocess_user_input(terms):
    # Preprocess user input, keeping specific phrases as unique tokens
    # Lowercase, lemmatize, remove stop words, punctuation, and non-alphabetic characters
    processed_terms = [preprocess(term) for term in terms]
    return " ".join(processed_terms)


def custom_tokenizer(text):
    # Directly split on commas if your inputs are assured to be well-formed
    return text.split(', ')


'''preprocessor=preprocess,
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        max_df=0.7,  # Termenii care apar în mai mult de 50% din documente sunt ignorați
        min_df=3,  # Termenii care apar în mai puțin de 3 documente sunt ignorați
        use_idf=True,
        norm='l2'
        '''


def create_tfidf_vectorizer(docs):
    vectorizer = TfidfVectorizer(
        max_df=0.7, max_features=25, min_df=3, ngram_range=(1, 1), norm='l2', stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix


def calculate_similarity(combined_features_user, combined_features_docs):
    similarity_scores = cosine_similarity(combined_features_user, combined_features_docs)
    return similarity_scores


def compare_user_input_tfidf(vectorizer, user_input, tfidf_matrix):
    user_tfidf = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)
    return similarity_scores


def compare_user_input_embeddings(reviews_docs, user_input):
    doc_embeddings = get_embeddings(reviews_docs)
    user_input_embedding = get_embeddings([user_input])
    embedding_scores = cosine_similarity(user_input_embedding, doc_embeddings)
    return embedding_scores


from sklearn.metrics import euclidean_distances


def compare_user_input_euclidean(vectorizer, user_input, tfidf_matrix):
    user_tfidf = vectorizer.transform([user_input])
    similarity_scores = 1 / (1 + euclidean_distances(user_tfidf, tfidf_matrix))
    return similarity_scores


from sentence_transformers import SentenceTransformer

# load the model from SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')


def get_embeddings(docs):
    return model.encode(docs, show_progress_bar=True)


def combine_features(tfidf_matrix, embeddings):
    # Ensure the matrices are in dense format if TF-IDF is sparse
    tfidf_dense = tfidf_matrix.toarray() if hasattr(tfidf_matrix, "toarray") else tfidf_matrix
    combined_features = np.concatenate((tfidf_dense, embeddings), axis=1)
    return combined_features


def calculate_similarity(combined_features_user, combined_features_docs):
    similarity_scores = cosine_similarity(combined_features_user, combined_features_docs)
    return similarity_scores


def combined_similarity_score(tfidf_scores, embedding_scores, alpha=0.9):
    # Combinați scorurile TF-IDF și embeddings, ajustând ponderile fiecărui set de scoruri
    combined_scores = alpha * tfidf_scores + (1 - alpha) * embedding_scores
    return combined_scores
