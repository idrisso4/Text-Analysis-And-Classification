from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import re
import pickle
import bert
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


def preprocess_text(text,stem=False):
    # Lower case
    text = text.lower().strip()
    # Removing html tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove punctuations and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Removing stop words
    text =  [word.lower() for word in text.split() if word not in stopwords.words('english')]
    # lemmatize data
    stemmer = WordNetLemmatizer()
    text = [stemmer.lemmatize(word) for word in text]
    vector = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(text)))
    return vector

f = open("tokenizer", "rb")
tokenizer = pickle.load(f)
f.close()

text_model = tf.keras.models.load_model("model")

categories = ['BUSINESS', 'CRIME', 'POLITICS', 'SCIENCE', 'SPORTS', 'TECH']


@app.route('/', methods=['POST'])
def post_review():
    stop_words = stopwords.words('english')
    input_text= request.form['input_text'].lower()
    
    processed_doc1 = ' '.join([word for word in input_text.split() if word not in stop_words])

    
    a=text_model.predict([preprocess_text(input_text)])
    category = categories[np.argmax(a)]
    print(a*100)
    print(category)

    sia = SentimentIntensityAnalyzer()
    ps = sia.polarity_scores(text=processed_doc1)
    print(ps)
    l = list(ps.values())[:-1]
    ps['max'] = l.index(max(l))
    ps['neu'] = round(ps['neu']*100,2)
    ps['neg'] = round(ps['neg']*100,2)
    ps['pos'] = round(ps['pos']*100,2)
    ps['catego'] = category
    ps['cat_val'] = round(np.max(a)*100,2)

    return render_template('home.html', final=ps , text=input_text)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
