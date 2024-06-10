import string

import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Load Spacy model
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

# Exemplu de recenzie
review = "Amazing Tour. Me and my sister took this tour with Dayna. We had so much fun despite the freezing cold. You get to see a lot of sites you wouldn't normally see if you went on your own. Dayna was hilarious and kept the tour fun."

# Lowercase
lowercase_review = review.lower()

# Cleaning
cleaned_review = lowercase_review.translate(str.maketrans('', '', string.punctuation))

# Tokenization
tokenized_review = cleaned_review.split()

# Stop-words removal
filtered_review = [word for word in tokenized_review if word not in stopwords.words('english')]

# Lemmatization using Spacy
doc = nlp(" ".join(filtered_review))
lemmatized_review = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

# Crearea DataFrame-ului pentru ilustrare
data = {
    'Step': ['DOCUMENT', 'LOWERCASE', 'CLEANING', 'TOKENIZATION', 'STOP-WORDS', 'LEMMATIZATION'],
    'Content': [
        review,
        lowercase_review,
        cleaned_review,
        tokenized_review,
        filtered_review,
        lemmatized_review
    ]
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
        width: 40%;
        border-collapse: collapse;
    }}
    th, td {{
        padding: 8px;
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
with open("text_processing_steps_nltk_spacy.html", "w") as file:
    file.write(html_content)
# Salvarea tabelului într-un fișier HTML pentru a-l vizualiza frumos
#df.to_html("text_processing_steps_nltk_spacy.html", index=False)
