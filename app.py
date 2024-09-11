from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data['message']
    
    # Transform the message using the vectorizer
    transformed_message = vectorizer.transform([message])
    
    # Predict using the loaded model
    prediction = model.predict(transformed_message)[0]
    
    return jsonify({'prediction': 'spam' if prediction == 1 else 'non-spam'})

if __name__ == "__main__":
    app.run(debug=True)
