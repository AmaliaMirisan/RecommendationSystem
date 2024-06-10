import joblib
from sklearn.metrics.pairwise import cosine_similarity
from TFIDFmodel.LoadData import load_data, load_json_data_for_links, filter_attractions, filter_attractions_cluster
from TFIDFmodel.AI import compare_user_input_tfidf, create_tfidf_vectorizer, preprocess_user_input, combined_similarity_score, compare_user_input_embeddings
from TFIDFmodel.Print import find_link, normalize_name

# Load necessary data and models
attractions_links = load_json_data_for_links('C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_cat.json')  # Update the path as necessary
details_files = ['C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_details_batch1.json', 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_details_batch2.json']
reviews_files = ['C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_reviews_batch1.json', 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\datasets\\attractions_reviews_batch2.json']
all_attractions = load_data(details_files, reviews_files)
vectorizer_path = 'C:\\Users\\miris\\Desktop\\lic\\recommendationSystem\\AdvancedTuningTF-IDF\\best_tfidf_model.joblib'
vectorizer = joblib.load(vectorizer_path)

def generate_recommendations(user_responses, user_city, user_budget):
    # create a list from all answers, except for "city" and "budget"
    user_input_list = [response.strip() for key, response in user_responses.items() if key not in ["city", "budget"]]
    user_input_processed = preprocess_user_input(user_input_list)
    filtered_attractions, filtered_no_reviews = filter_attractions(all_attractions, user_city, user_budget)
    reviews_docs = [" ".join(attraction['reviews']) for attraction in filtered_attractions if 'reviews' in attraction]
    tfidf_matrix = vectorizer.transform(reviews_docs)
    tfidf_scores = compare_user_input_tfidf(vectorizer, user_input_processed, tfidf_matrix)
    embedding_scores = compare_user_input_embeddings(reviews_docs, user_input_processed)
    combined_scores = combined_similarity_score(tfidf_scores, embedding_scores)
    top_indices = combined_scores.argsort()[0][::-1]

    recommendations = []
    for idx in top_indices[:5]:  # Top 5 recommendations
        attraction = filtered_attractions[idx]
        processed_name = normalize_name(attraction['name'])
        link = find_link(attractions_links, processed_name)
        recommendations.append({
            "name": normalize_name(attraction['name']),
            "location": attraction['city'],
            "price": attraction.get('price', 'Not available'),
            "link": link if link != 'No link found' else 'Link not available',
            "score": combined_scores[0][idx]
        })
    return recommendations