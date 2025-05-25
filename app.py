from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import os
import pickle


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def custom_preprocessor(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    return text.lower()

def custom_tokenizer(text):
    words = text.split()
    return [ps.stem(w) for w in words if w not in stop_words]

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and vectorizer
with open('book_review_svm.pkl', 'rb') as f:
    saved = pickle.load(f)
    model = saved['model']
    vectorizer = saved['vectorizer']

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        abort(400, description="Request must be JSON")

    data = request.get_json()
    if 'comment' not in data:
        abort(400, description="JSON must contain 'comment'")

    raw_text = data['comment']
    if not isinstance(raw_text, str) or raw_text.strip() == "":
        abort(400, description="'comment' must be a non-empty string")

    X_new = vectorizer.transform([raw_text]).toarray()
    pred = model.predict(X_new)[0]
    label = "Positive" if pred == 1 else "Negative"

    return jsonify({"sentiment": label})

if __name__ == '__main__':
    # Get PORT from environment (for production), fallback to 3000 for local dev
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
