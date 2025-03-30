# SaaS Product Reviews Sentiment Dashboard & Classifier

This project analyzes customer reviews of SaaS products using sentiment analysis.  
It uses real-world review data from **Amazon** and **Trustpilot**, compares two machine learning models, and lets users try live predictions using the best model — all through an interactive Streamlit dashboard.

---

## ✨ Features

- 📊 **Dashboard View**:
  - Sentiment distribution bar chart
  - Word clouds per sentiment (Positive, Neutral, Negative)
  - Model comparison: Logistic Regression vs Naive Bayes
  - Classification reports and confusion matrices

- 🧠 **Live Prediction View**:
  - Type in your own review
  - Get instant sentiment prediction using the best model
  - See the model’s confidence in the prediction

---

## 📁 Project Structure

📦 saas-review-model/ ├── app.py # Streamlit dashboard + prediction app ├── cleaned_combined_reviews.csv # Preprocessed review dataset (reduced) ├── model_results.json # Evaluation results for both models ├── tfidf_vectorizer.pkl # Trained TF-IDF vectorizer ├── logreg_model.pkl # Trained Logistic Regression model ├── label_encoder.pkl # Label encoder used for predictions ├── requirements.txt # Dependencies └── README.md # You're reading it


---

## 🤖 Models Used

- **Logistic Regression** with class weights (`class_weight='balanced'`) ✅ Used for live predictions
- **Multinomial Naive Bayes** for comparison

Both models were trained on a labeled version of the review dataset and evaluated using:
- Precision, Recall, F1-score
- Confusion matrix
- Macro & Weighted Averages

---

## 🚀 How to Run the App

### 🔗 Option 1: [Streamlit Cloud](https://share.streamlit.io) (Recommended)

1. Fork this repo or upload your own
2. Go to: [https://share.streamlit.io](https://share.streamlit.io)
3. Click **“New App”**
4. Choose this repo → main branch → `app.py`
5. Click **Deploy**

Your app will go live instantly 🎉

---

### 💻 Option 2: Run Locally

1. Clone this repo
2. Install dependencies:

```bash
pip install -r requirements.txt

Run the app:
streamlit run app.py

📓 Data Sources
Amazon Fine Food Reviews (via Kaggle)

Trustpilot Reviews (2022 snapshot)

Combined and labeled with sentiment scores

👤 Author
Abdalla Soub
University of Sunderland – MSc Data Science
Module: CETM46 – Data Science Product Development
