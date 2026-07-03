import streamlit as st
import pickle
import re
import os

# 1. Simple text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    return text

# 2. Dynamic path configuration handling the extra space in file name
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'naive_bayes_model .pkl') # Space included to match explorer
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

@st.cache_resource
def load_assets():
    with open(model_path, 'rb') as f_model:
        model = pickle.load(f_model)
    with open(vectorizer_path, 'rb') as f_vec:
        vectorizer = pickle.load(f_vec)
    return model, vectorizer

# 3. Streamlit Page Layout
st.set_page_config(page_title="IMDB Sentiment App", page_icon="🎬")
st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write("Enter your movie review below to check whether the sentiment is Positive or Negative.")

try:
    # Load the serialized assets
    model, vectorizer = load_assets()
    
    # User text input box
    user_review = st.text_area("Review Content:", placeholder="Type your movie review here...")

    if st.button("Predict Sentiment"):
        if user_review.strip() == "":
            st.warning("Please enter some text first!")
        else:
            # Process and predict
            cleaned_text = clean_text(user_review)
            vectorized_input = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_input)
            
            # Display result
            st.subheader("Final Prediction:")
            if prediction == 1 or str(prediction).lower() == 'positive' or str(prediction).lower() == 'pos':
                st.success("🎉 **Positive Sentiment!** The viewer liked the movie.")
            else:
                st.error("🚨 **Negative Sentiment!** The viewer disliked the movie.")

except Exception as e:
    st.error(f"⚠️ Error loading files: {e}")
