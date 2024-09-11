import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Spam Detection Dashboard")

# User Input
user_input = st.text_area("Enter the message:")

if st.button('Predict'):
    # Transform the user input
    transformed_input = vectorizer.transform([user_input])
    
    # Predict using the model
    prediction = model.predict(transformed_input)[0]
    result = "Spam" if prediction == 1 else "Non-Spam"
    
    st.write(f"The message is: **{result}**")

# Visualization (Example: Word Cloud for Spam)
df = pd.read_csv("/content/drive/MyDrive/spam.csv", encoding='ISO-8859-1')
df.columns = ['Spam', 'Message']
spam_messages = ' '.join(df[df['Spam'] == 'spam']['Message'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_messages)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot()
