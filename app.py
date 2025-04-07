# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import joblib
import re

# --- Load model & vectorizer ---
model = joblib.load("xgboost_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --- Load dataset for dashboard ---
df = pd.read_csv("cleaned_combined_reviews.csv")

# --- Cleaning functions (for live prediction) ---
stop_words = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
    'it','its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were','be',
    'been','being','have','has','had','having','do','does','did','doing','a','an',
    'the','and','but','if','or','because','as','until','while','of','at','by','for',
    'with','about','against','between','into','through','during','before','after',
    'above','below','to','from','up','down','in','out','on','off','over','under',
    'again','further','then','once','here','there','when','where','why','how','all',
    'any','both','each','few','more','most','other','some','such','no','nor','not',
    'only','own','same','so','than','too','very','can','will','just','don','should','now'
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

# --- Streamlit Layout ---
st.set_page_config(page_title="SaaS Sentiment Dashboard", layout="wide")
st.title("📊 SaaS Product Review Sentiment Analysis")
st.markdown("Welcome! This app allows you to predict sentiment for any review and explore dashboard insights based on real Amazon/Trustpilot reviews.")

# Sidebar toggle
mode = st.sidebar.radio("Choose View", ["Live Prediction", "Dashboard"])

# === LIVE PREDICTION ===
if mode == "Live Prediction":
    st.subheader("🔮 Live Sentiment Prediction")
    st.markdown("✏️ **How to use:** Enter a review below (e.g. _'The software is easy to use and powerful.'_) then click Predict.")

    user_input = st.text_area("📝 Enter your SaaS review:")

    if st.button("🔍 Predict Sentiment"):
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

            color_map = {
                "Positive": "green",
                "Neutral": "orange",
                "Negative": "red"
            }

            st.markdown(f"<h3 style='color:{color_map[label]}'>✅ Sentiment: {label}</h3>", unsafe_allow_html=True)
            st.markdown(f"🔍 <b>Confidence:</b> `{confidence:.2%}`", unsafe_allow_html=True)

# === DASHBOARD ===
elif mode == "Dashboard":
    st.subheader("📊 Dashboard Overview")

    try:
        with open("model_results.json", "r") as f:
            results = json.load(f)
    except:
        st.error("Model results not found. Please upload 'model_results.json'.")
        st.stop()

    model_choice = st.sidebar.selectbox("Select Model to View Results", list(results.keys()))

    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "orange", "red"])
    ax1.set_title("Sentiment Counts")
    ax1.set_ylabel("Number of Reviews")
    st.pyplot(fig1)

    # Pie chart
    st.subheader("Sentiment Proportion (Pie Chart)")
    fig2, ax2 = plt.subplots()
    ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct="%1.1f%%",
            colors=["green", "orange", "red"], startangle=90)
    st.pyplot(fig2)

    # Source comparison chart
    st.subheader("Sentiment by Source")
    source_counts = df.groupby(["source", "sentiment"]).size().unstack().fillna(0)
    source_counts.plot(kind="bar", stacked=True, color=["blue", "lightskyblue", "red"])
    st.pyplot(plt.gcf())

    # Word clouds
    st.subheader("Word Clouds by Sentiment")
    col1, col2, col3 = st.columns(3)
    for sentiment, col in zip(["Positive", "Neutral", "Negative"], [col1, col2, col3]):
        text = " ".join(df[df["sentiment"] == sentiment]["clean_text"])
        wordcloud = WordCloud(width=300, height=200, background_color="white").generate(text)
        col.image(wordcloud.to_array(), caption=sentiment)

    # Top keywords
    st.subheader("🧠 Top Keywords by Sentiment")
    for sentiment in ["Positive", "Neutral", "Negative"]:
        st.markdown(f"**{sentiment} Reviews:**")
        subset = df[df["sentiment"] == sentiment]
        all_words = " ".join(subset["clean_text"]).split()
        keywords = pd.Series([w for w in all_words if w not in stop_words]).value_counts().head(10)
        st.dataframe(pd.DataFrame({"Keyword": keywords.index, "Count": keywords.values}))

    # Model Comparison
    st.subheader(f"📈 Model Performance - {model_choice}")
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
    st.caption("Built with Streamlit | Models: Logistic Regression, XGBoost | Data: Amazon & Trustpilot")
