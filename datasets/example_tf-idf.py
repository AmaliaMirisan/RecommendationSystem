import string
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Spacy model
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

# Exemplu de recenzie
review = "Amazing Tour. Me and my sister took this tour with Dayna. We had so much fun despite the freezing cold. You get to see a lot of sites you wouldn't normally see if you went on your own. Dayna was hilarious and kept the tour fun."

# Documente de recenzii (un exemplu simplificat)
reviews_docs = [
    "Amazing Tour. Me and my sister took this tour with Dayna. We had so much fun despite the freezing cold. You get to see a lot of sites you wouldn't normally see if you went on your own. Dayna was hilarious and kept the tour fun.",
    "Great experience. I enjoyed the sights and the guide was very knowledgeable.",
    "It was an okay tour. Nothing special but the guide was friendly."
]

# Preprocesare text
def preprocess(text):
    # Lowercase, remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = text.split()
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatization
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return lemmatized

# Preprocesarea recenziilor
processed_reviews = [" ".join(preprocess(doc)) for doc in reviews_docs]

# Vectorizarea TF-IDF
vectorizer = TfidfVectorizer(norm='l2', max_df=0.7, min_df=1, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(processed_reviews)

# Obținerea scorurilor TF-IDF pentru recenzia exemplificată
tfidf_scores = tfidf_matrix.toarray()[0]  # Scorurile pentru primul document

# Crearea DataFrame-ului pentru ilustrare
feature_names = vectorizer.get_feature_names_out()
data = {
    'Term': feature_names,
    'TF-IDF Score': tfidf_scores
}

df = pd.DataFrame(data)
print(df)

# Salvarea tabelului într-un fișier HTML cu stiluri CSS pentru lățimea coloanelor
html_content = df.to_html(index=False)

# Adăugare stiluri CSS pentru a face tabelul mai lat
html_content = f"""
<html>
<head>
<style>
    table {{
        width: 20%;
        border-collapse: collapse;
    }}
    th, td {{
        padding: 5px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }}
    th {{
        background-color: #f2f2f2;
    }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Salvare fișier HTML
with open("tfidf_scores.html", "w") as file:
    file.write(html_content)
