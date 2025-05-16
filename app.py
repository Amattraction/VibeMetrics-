from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
with open('model_and_vectorizer.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

# Preprocessing function
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join([port_stem.stem(word) for word in text if word not in stop_words])

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['tweet']
    processed = preprocess(user_input)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)

    result = "Positive" if prediction[0] == 1 else "Negative "
    return render_template('index.html', prediction=result, tweet=user_input)

# Run the app
if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)
