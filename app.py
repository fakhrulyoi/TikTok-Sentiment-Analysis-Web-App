from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

# Get absolute path to current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model and vectorizer paths (relative to project root)
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

try:
    # Load the model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or vectorizer: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            text = request.form['review']
            # Transform the text using the saved vectorizer
            text_transformed = vectorizer.transform([text])
            # Make prediction
            prediction = model.predict(text_transformed)[0]
            # Get prediction probability
            probabilities = model.predict_proba(text_transformed)[0]
            confidence = max(probabilities) * 100
            
            # Get emoji based on sentiment
            emoji = "😊" if prediction == "Positive" else "😐" if prediction == "Neutral" else "😔"
            
            return jsonify({
                'sentiment': prediction,
                'confidence': f"{confidence:.2f}%",
                'emoji': emoji,
                'success': True
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
