import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# TEXT PREPROCESSING FUNCTION
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# -----------------------------
# IMPROVED TRAINING DATA
# -----------------------------
data = {
    "text": [
        "i want refund",
        "payment failed",
        "app is not working",
        "need help with account",
        "charged twice",
        "bug in application",
        "refund not received",
        "unable to login",
        "transaction failed",
        "error in system",
        "account locked",
        "payment not processed"
    ],
    "category": [
        "Refund",
        "Payment",
        "Technical",
        "Support",
        "Payment",
        "Technical",
        "Refund",
        "Support",
        "Payment",
        "Technical",
        "Support",
        "Payment"
    ]
}

df = pd.DataFrame(data)
df["text"] = df["text"].apply(clean_text)

# -----------------------------
# MODEL TRAINING
# -----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

model = MultinomialNB()
model.fit(X, df["category"])

# -----------------------------
# UI DESIGN
# -----------------------------
st.set_page_config(page_title="Smart Ticket Classifier", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center;'>🤖 Smart Support Ticket Classifier</h1>
    <p style='text-align: center; color: gray;'>
    AI-powered system to classify support tickets & assign priority automatically.
    </p>
    """,
    unsafe_allow_html=True
)

# INPUTS
ticket = st.text_area("📌 Enter Ticket Description", height=150)
product = st.text_input("📦 Product Name (Optional)")

# -----------------------------
# CLASSIFICATION BUTTON
# -----------------------------
if st.button("🚀 Classify Ticket"):

    if ticket:
        cleaned = clean_text(ticket)
        transformed = vectorizer.transform([cleaned])

        prediction = model.predict(transformed)[0]
        confidence = model.predict_proba(transformed).max()

        # -----------------------------
        # PRIORITY LOGIC (IMPROVED)
        # -----------------------------
        if prediction == "Refund":
            priority = "🔴 CRITICAL"
        elif prediction == "Payment":
            priority = "🟠 HIGH"
        elif prediction == "Technical":
            priority = "🟡 MEDIUM"
        else:
            priority = "🟢 LOW"

        st.success("✅ Ticket processed successfully!")

        st.markdown("---")
        st.markdown("## 📊 Classification Results")

        col1, col2 = st.columns(2)

        # CATEGORY CARD
        with col1:
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#111827;">
                    <h4 style="color:white;">📂 Category</h4>
                    <p style="font-size:20px; color:#38bdf8;"><b>{prediction}</b></p>
                    <p style="color:gray;">Confidence: {confidence:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # PRIORITY CARD
        with col2:
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#111827;">
                    <h4 style="color:white;">⚡ Priority</h4>
                    <p style="font-size:20px; color:#f87171;"><b>{priority}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # EXTRA INFO
        if product:
            st.info(f"📦 Related to product: **{product}**")

    else:
        st.error("⚠️ Please enter ticket description")