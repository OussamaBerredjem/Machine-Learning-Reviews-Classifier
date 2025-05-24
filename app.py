from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import pickle
from text_preprocessing import custom_preprocessor, custom_tokenizer  # Required for unpickling

app = Flask(__name__)
CORS(app)  # Allow all origins (you can restrict this)

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
    app.run(host='127.0.0.1', port=3000, debug=True)
