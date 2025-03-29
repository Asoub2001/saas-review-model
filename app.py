# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json

# Load dataset
df = pd.read_csv("https://drive.google.com/uc?export=download&id=YOUR_FILE_ID")

# Load model results
with open("model_results.json", "r") as f:
    results = json.load(f)

# Sidebar: Model Selector
model_choice = st.sidebar.selectbox("Select Model to View Results", list(results.keys()))

st.title("SaaS Product Reviews Sentiment Dashboard")
st.write("Compare sentiment trends and model performance across platforms like Amazon and Trustpilot.")

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
st.subheader("Model Performance - {}".format(model_choice))
model_data = results[model_choice]

st.text("Classification Report")
st.text(model_data["report"])

# Plot confusion matrix if available
if "conf_matrix" in model_data:
    conf_matrix = pd.DataFrame(model_data["conf_matrix"],
                               index=["Negative", "Neutral", "Positive"],
                               columns=["Predicted Negative", "Predicted Neutral", "Predicted Positive"])
    st.subheader("Confusion Matrix")
    st.dataframe(conf_matrix)

st.markdown("---")
st.caption("Built with Streamlit | Data from Amazon & Trustpilot | Models: Logistic Regression, Naive Bayes")
