import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from LoadData import load_data, load_json_data_for_links

# load the model from SentenceTransformer
embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')

# load data
details_files = ['C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_details_batch1.json', 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_details_batch2.json']
reviews_files = ['C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_reviews_batch1.json', 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_reviews_batch2.json']

all_attractions = load_data(details_files, reviews_files)

# process the reviews and compute embeddings
reviews_docs = [" ".join(attraction['reviews']) for attraction in all_attractions if 'reviews' in attraction]
embeddings = embedding_model.encode(reviews_docs, show_progress_bar=True)

# save embeddings and attractions list
np.save('C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\TFIDFmodel\\embeddings.py.npy', embeddings)
joblib.dump(all_attractions, 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\TFIDFmodel\\embeddingsModel.joblib')
