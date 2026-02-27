import streamlit as st
import pickle
import re

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

st.title("Fake News Detection System")

st.write("Enter a news article below to check whether it is Fake or Real.")

input_text = st.text_area("News Article")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(input_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 0:
            st.error("This News is FAKE!!")
        else:
            st.success("This News is REAL")