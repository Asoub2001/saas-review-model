# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import joblib
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# --- Load model & vectorizer ---
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("logreg_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")  # Optional if you're using encoded labels

try:
    df = pd.read_csv("cleaned_combined_reviews.csv")
    if "sentiment" not in df.columns:
        st.error("❌ The dataset did not load correctly. Column 'sentiment' not found.")
        st.stop()
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

    

# --- Cleaning functions ---
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

# --- Streamlit App Layout ---
st.set_page_config(page_title="SaaS Sentiment Dashboard", layout="wide")
st.title("📊 SaaS Product Reviews Sentiment App")
st.markdown("Toggle between dashboard insights and a live prediction tool.")

# --- Sidebar Toggle ---
mode = st.sidebar.radio("Choose Mode", ["Dashboard", "Live Prediction"])

if mode == "Dashboard":
    # Load model results from JSON
    try:
        with open("model_results.json", "r") as f:
            results = json.load(f)
    except:
        st.error("Model results not found. Please upload 'model_results.json'.")
        st.stop()

    model_choice = st.sidebar.selectbox("Select Model to View Results", list(results.keys()))

    # Section 1: Sentiment Distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "orange", "red"])
    ax1.set_title("Sentiment Counts")
    ax1.set_ylabel("Number of Reviews")
    st.pyplot(fig1)

    # Section 2: Word Clouds by Sentiment
    st.subheader("Word Clouds by Sentiment")
    col1, col2, col3 = st.columns(3)
    for sentiment, col in zip(["Positive", "Neutral", "Negative"], [col1, col2, col3]):
        text = " ".join(df[df["sentiment"] == sentiment]["clean_text"])
        wordcloud = WordCloud(width=300, height=200, background_color="white").generate(text)
        col.image(wordcloud.to_array(), caption=sentiment)

    # Section 3: Model Comparison
    st.subheader(f"Model Performance - {model_choice}")
    model_data = results[model_choice]
    st.text("Classification Report")
    st.text(model_data["report"])

    if "conf_matrix" in model_data:
        conf_matrix = pd.DataFrame(model_data["conf_matrix"],
                                   index=["Negative", "Neutral", "Positive"],
                                   columns=["Predicted Negative", "Predicted Neutral", "Predicted Positive"])
        st.subheader("Confusion Matrix")
        st.dataframe(conf_matrix)

    st.markdown("---")
    st.caption("Built with Streamlit | Models: Logistic Regression, Naive Bayes")

elif mode == "Live Prediction":
    st.subheader("🧠 Try the Sentiment Classifier!")
    user_input = st.text_area("Enter a customer review:", height=150)

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            cleaned = clean_text(user_input)
            cleaned = remove_stopwords(cleaned)
            vect_input = vectorizer.transform([cleaned])
            prediction = model.predict(vect_input)[0]
            probs = model.predict_proba(vect_input)[0]
            confidence = max(probs)
            label = label_encoder.inverse_transform([prediction])[0]

            st.success(f"✅ Predicted Sentiment: **{label}**")
            st.markdown(f"🔍 **Confidence**: `{confidence:.2%}`")
