# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import joblib
import re
from collections import Counter

# --- Hardcoded English stopwords ---
stop_words = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves',
    'he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their',
    'theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an',
    'the','and','but','if','or','because','as','until','while','of','at','by','for','with','about',
    'against','between','into','through','during','before','after','above','below','to','from','up',
    'down','in','out','on','off','over','under','again','further','then','once','here','there','when',
    'where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor',
    'not','only','own','same','so','than','too','very','can','will','just','don','should','now'
])

# --- Load model & vectorizer ---
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("logreg_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --- Load dataset ---
try:
    df = pd.read_csv("cleaned_combined_reviews.csv")
    if "sentiment" not in df.columns:
        st.error("❌ Dataset missing 'sentiment' column.")
        st.stop()
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# --- Cleaning functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

# --- Streamlit layout ---
st.set_page_config(page_title="SaaS Review Sentiment App", layout="wide")

# --- Sidebar Navigation ---
mode = st.sidebar.radio("Choose View", ["Live Prediction", "Dashboard"])

# --- About the App ---
st.title("📊 SaaS Product Reviews Sentiment App")
st.markdown("""
Welcome to the **SaaS Sentiment Analysis App**!  
This app lets you:
- 🧠 **Predict sentiment** of any SaaS customer review using a trained machine learning model
- 📈 **Visualize and analyze** reviews from Amazon and Trustpilot
- 🤖 Uses **Logistic Regression with balanced class weights** for better real-world performance  
""")

# --- Live Prediction ---
if mode == "Live Prediction":
    st.header("🧠 Live Sentiment Prediction")
    st.markdown("""
**How to use:**  
Type a customer review below (e.g. *"This software is easy to use and really powerful"*)  
Click **Predict** to see the sentiment classification.
""")
    user_input = st.text_area("✍️ Enter your SaaS review:", height=150)

    if st.button("🔍 Predict Sentiment"):
        if not user_input.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned = remove_stopwords(clean_text(user_input))
            vect_input = vectorizer.transform([cleaned])
            prediction = model.predict(vect_input)[0]
            probs = model.predict_proba(vect_input)[0]
            confidence = max(probs)
            label = label_encoder.inverse_transform([prediction])[0]

            st.success(f"✅ Predicted Sentiment: **{label}**")
            st.markdown(f"🔍 **Confidence:** `{confidence:.2%}`")

# --- Dashboard ---
elif mode == "Dashboard":
    st.header("📈 Dashboard: SaaS Reviews Analysis")

    # Load model results
    try:
        with open("model_results.json", "r") as f:
            results = json.load(f)
    except:
        st.error("Missing 'model_results.json' file.")
        st.stop()

    model_choice = st.sidebar.selectbox("🔬 Select Model to View Results", list(results.keys()))

    # 1. Sentiment Distribution
    st.subheader("🔢 Overall Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "orange", "red"])
    ax1.set_ylabel("Number of Reviews")
    st.pyplot(fig1)

    # 2. Sentiment by Source
    st.subheader("📊 Sentiment Breakdown by Source")
    sentiment_by_source = df.groupby(['source', 'sentiment']).size().unstack().fillna(0)
    st.bar_chart(sentiment_by_source)

    # 3. Pie Chart
    st.subheader("🥧 Sentiment Proportions")
    fig2, ax2 = plt.subplots()
    ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["green", "orange", "red"])
    ax2.axis('equal')
    st.pyplot(fig2)

    # 4. Word Clouds by Sentiment
    st.subheader("☁️ Word Clouds by Sentiment")
    col1, col2, col3 = st.columns(3)
    for sentiment, col in zip(["Positive", "Neutral", "Negative"], [col1, col2, col3]):
        text = " ".join(df[df["sentiment"] == sentiment]["clean_text"])
        wordcloud = WordCloud(width=300, height=200, background_color="white").generate(text)
        col.image(wordcloud.to_array(), caption=sentiment)

    # 5. Top Keywords by Sentiment
    st.subheader("🧠 Top Keywords by Sentiment")
    for sentiment in ["Positive", "Neutral", "Negative"]:
        st.markdown(f"**{sentiment} Reviews:**")
        all_words = " ".join(df[df["sentiment"] == sentiment]["clean_text"]).split()
        keywords = [word for word in all_words if word not in stop_words]
        freq = Counter(keywords).most_common(10)
        freq_df = pd.DataFrame(freq, columns=["Keyword", "Count"])
        st.dataframe(freq_df)

    # 6. Model Performance
    st.subheader(f"📋 Model Evaluation – {model_choice}")
    model_data = results[model_choice]
    st.text("Classification Report")
    st.text(model_data["report"])

    if "conf_matrix" in model_data:
        conf_matrix = pd.DataFrame(model_data["conf_matrix"],
                                   index=["Negative", "Neutral", "Positive"],
                                   columns=["Pred. Negative", "Pred. Neutral", "Pred. Positive"])
        st.subheader("📊 Confusion Matrix")
        st.dataframe(conf_matrix)

    st.markdown("---")
    st.caption("Built with Streamlit | Data: Amazon + Trustpilot | Models: Logistic Regression, Naive Bayes")
