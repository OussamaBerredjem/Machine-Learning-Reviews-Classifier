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
