import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
from flask import Flask, render_template, request, session
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import yake

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load model and vectorizer
try:
    with open('model_and_vectorizer.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("Error: 'model_and_vectorizer.pkl' file not found.")
    raise
except Exception as e:
    print(f"Error loading model and vectorizer: {e}")
    raise

# Preprocessing setup
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))
# Remove negation words from stopwords to preserve them
negation_words = {'not', 'never', 'no', 'neither', 'nor'}
stop_words = stop_words - negation_words  # Exclude negation words

def preprocess(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)  # Remove URLs, mentions, hashtags
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    # Preserve negation words and stem other words
    processed_words = []
    for i, word in enumerate(text):
        if word in negation_words:
            processed_words.append(word)  # Keep negation words as-is
        elif word not in stop_words and len(word) > 2:
            processed_words.append(port_stem.stem(word))
    return ' '.join(processed_words)

def extract_keywords(text, max_keywords=10):
    try:
        if not text or not isinstance(text, str):
            return []
        kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=max_keywords)
        keywords = kw_extractor.extract_keywords(text)
        return [(kw, score) for kw, score in keywords]
    except Exception as e:
        print(f"Error in keyword extraction: {e}")
        return []

def generate_sentiment_graph(history):
    pos_count = sum(1 for _, pred in history if "Positive" in pred)
    neg_count = sum(1 for _, pred in history if "Negative" in pred)
    neutral_count = len(history) - pos_count - neg_count  # In case future logic adds neutral
    
    labels = ['Positive', 'Negative', 'Neutral']
    counts = [pos_count, neg_count, neutral_count]
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color=['green', 'red', 'gray'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')
    plt.title(f'Sentiment Trends (Last {len(history)} Searches)')
    plt.ylim(0, max(counts) + 1 if counts else 1)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    tweet = ""
    sentiment_score = None
    loading = False
    keywords = []

    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        tweet = request.form.get('tweet', '')
        if tweet:
            processed = preprocess(tweet)
            vectorized = vectorizer.transform([processed])
            pred = model.predict(vectorized)
            # Initial prediction
            prediction = "😄 Positive" if pred[0] == 1 else "😞 Negative"
            sentiment_score = 1 if pred[0] == 1 else -1
            
            # Rule-based negation check
            words = processed.split()
            for i, word in enumerate(words):
                if word in negation_words and i + 1 < len(words):
                    next_word = words[i + 1]
                    # If negation is followed by a positive-leaning word, flip the sentiment
                    positive_words = {'like', 'love', 'good', 'great', 'awesome', 'happy'}
                    if next_word in positive_words and "Positive" in prediction:
                        prediction = "😞 Negative"
                        sentiment_score = -1
                        break
            
            history = session['history']
            history.insert(0, (tweet, prediction))
            session['history'] = history[:25]
            session.modified = True
            loading = False

    graph_img = generate_sentiment_graph(session['history'])
    sample_results = [
        ("I love this sunny day!", "😄 Positive"),
        ("This is the worst experience ever.", "😞 Negative")
    ]

    return render_template(
        'index.html',
        prediction=prediction,
        tweet=tweet,
        history=session['history'],
        graph_img=graph_img,
        sentiment_score=sentiment_score,
        loading=loading,
        sample_results=sample_results,
        keywords=keywords
    )

@app.route('/wordcloud')
def show_wordcloud():
    if 'history' not in session or not session['history']:
        return render_template('wordcloud.html', word_cloud_img=None, tweet=None, message="No recent tweet available to display the word cloud.")
    
    # Get the most recent tweet from history
    recent_tweet, _ = session['history'][0]
    word_cloud_img = generate_word_cloud(recent_tweet)
    return render_template('wordcloud.html', word_cloud_img=word_cloud_img, tweet=recent_tweet, message=None)

def generate_word_cloud(text):
    try:
        if not text or not isinstance(text, str):
            return None
        
        # Preprocess text for word cloud (similar to existing preprocess but keep more words)
        text = re.sub(r'http\S+|@\w+|#\w+', '', text)  # Remove URLs, mentions, hashtags
        text = re.sub('[^a-zA-Z\s]', '', text)  # Keep letters and spaces
        text = text.lower()
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=stop_words,
            min_font_size=10,
            max_words=100
        ).generate(text)
        
        # Create plot
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Hide axes
        plt.tight_layout(pad=0)
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None
    
@app.route('/graph')
def show_graph():
    if 'history' not in session or not session['history']:
        return render_template('graph.html', graph_img=None, message="No history available to display the graph.")
    
    graph_img = generate_sentiment_graph(session['history'])
    return render_template('graph.html', graph_img=graph_img, message=None)

@app.route('/analyze_emotions', methods=['GET', 'POST'])
def analyze_emotions():
    emotion_counts = None
    input_text = ''
    word_cloud_img = None
    keywords = []

    if request.method == 'POST':
        input_text = request.form.get('text', '')
        if input_text:
            emotion_counts = detect_emotions(input_text)
            word_cloud_img = generate_word_cloud(input_text)
            keywords = extract_keywords(input_text)

    return render_template(
        'analyze_emotions.html',
        emotion_counts=emotion_counts,
        input_text=input_text,
        word_cloud_img=word_cloud_img,
        keywords=keywords
    )

def detect_emotions(text):
    emotions = ['happy', 'sad', 'anger', 'fear', 'surprise', 'love', 'disgust']
    return {emotion: text.lower().count(emotion) for emotion in emotions}

if __name__ == '__main__':
    print("Starting Flask server on port 5001...")
   
