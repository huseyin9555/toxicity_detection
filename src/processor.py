import re
import nltk
from nltk.corpus import stopwords

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Cleans the input text by removing URLs, special characters.
    """
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove stopwords
    words = [w for w in text.split() if w not in STOP_WORDS]
    
    return " ".join(words)