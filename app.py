import streamlit as st
import joblib
from pathlib import Path
import re

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# Load models and vectorizer
@st.cache_resource
def load_model():
    models_dir = Path(__file__).parent / "models"
    vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
    model = joblib.load(models_dir / "passive_aggressive_model.joblib")
    return vectorizer, model

# Text cleaning function (same as train.py)
DATELINE_RE = re.compile(r"^.{0,120}?\(reuters\)\s*-\s*", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACES_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """Lowercase, drop Reuters datelines, URLs, punctuation and digits."""
    text = DATELINE_RE.sub("", str(text))
    text = text.lower()
    text = text.replace("reuters", " ")
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()

# Prediction function
def predict(text, vectorizer, model):
    features = vectorizer.transform([clean_text(text)])
    label = model.predict(features)[0]
    score = float(model.decision_function(features)[0])
    return label, abs(score)

# Main app
def main():
    st.title("🔍 Automated Fake News Detector")
    st.markdown("**Detect fake news using TF-IDF + Passive-Aggressive Classifier**")
    st.markdown("---")
    
    # Load model
    vectorizer, model = load_model()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Detector", "📊 About Model", "❓ FAQ"])
    
    # TAB 1: Main Detector
    with tab1:
        st.subheader("Paste your article here:")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])
        
        article_text = ""
        
        if input_method == "Paste Text":
            article_text = st.text_area(
                "Enter article text:",
                height=300,
                placeholder="Paste the full article here (at least 2-3 paragraphs for best results)..."
            )
        else:
            uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
            if uploaded_file is not None:
                article_text = uploaded_file.read().decode("utf-8")
                st.text_area("File content:", value=article_text, height=300, disabled=True)
        
        # Predict button
        if st.button("🔍 Analyze Article", use_container_width=True):
            if len(article_text.strip()) < 10:
                st.warning("⚠️ Please enter at least 10 characters")
            else:
                with st.spinner("Analyzing..."):
                    prediction, confidence = predict(article_text, vectorizer, model)
                    
                    # Display result
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == "REAL":
                            st.success(f"✅ PREDICTION: **REAL NEWS**")
                            color = "#2a9d8f"
                        else:
                            st.error(f"❌ PREDICTION: **FAKE NEWS**")
                            color = "#e76f51"
                    
                    with col2:
                        st.metric("Confidence Score", f"{confidence:.2f}")
                    
                    st.markdown("---")
                    
                    # Confidence bar
                    st.subheader("Confidence Level:")
                    st.progress(min(confidence / 10, 1.0))
                    
                    # Interpretation
                    st.subheader("📌 What this means:")
                    if prediction == "REAL":
                        st.info(
                            "This article appears to have characteristics of **real news**.\n\n"
                            "However, this model detects **writing style and source patterns**, not factual accuracy. "
                            "For critical decisions, consult professional fact-checkers."
                        )
                    else:
                        st.warning(
                            "This article appears to have characteristics of **fake news**.\n\n"
                            "However, this model detects **writing style and source patterns**, not factual accuracy. "
                            "For critical decisions, consult professional fact-checkers."
                        )
                    
                    # Confidence interpretation
                    st.subheader("Confidence Interpretation:")
                    if confidence < 1.0:
                        st.info("🤔 **Low confidence** - Model is uncertain")
                    elif confidence < 3.0:
                        st.info("📊 **Medium confidence** - Prediction is reasonably confident")
                    else:
                        st.info("💪 **High confidence** - Model is very confident in this prediction")
    
    # TAB 2: Model Information
    with tab2:
        st.subheader("📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "98.38%")
        col2.metric("Precision", "98.17%")
        col3.metric("Recall", "98.83%")
        col4.metric("F1-Score", "98.50%")
        
        st.markdown("---")
        
        st.subheader("🔧 Model Details")
        st.write("""
        - **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Classifier:** Passive-Aggressive Classifier (PAC)
        - **Training Data:** ISOT Fake News Dataset (38,820 articles)
        - **Vocabulary Size:** 95,819 unique words
        - **Training Time:** ~0.75 seconds
        - **Prediction Time:** ~0.0014 ms per article
        """)
        
        st.markdown("---")
        
        st.subheader("📈 Dataset Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Articles", "38,820")
        col2.metric("Real News", "20,921 (53.9%)")
        col3.metric("Fake News", "17,899 (46.1%)")
        
        st.markdown("---")
        
        st.subheader("⚠️ Important Limitations")
        st.warning("""
        1. **Detects style, not truth** - Model learns writing patterns, not factual accuracy
        2. **Domain-specific** - Trained on 2016-2017 US political news
        3. **Cross-dataset accuracy drops to 57.9%** - Shows generalization weakness
        4. **No context understanding** - Cannot understand word order or sarcasm
        5. **English only** - May not work for other languages
        """)
    
    # TAB 3: FAQ
    with tab3:
        st.subheader("❓ Frequently Asked Questions")
        
        with st.expander("What does TF-IDF do?"):
            st.write("""
            TF-IDF (Term Frequency-Inverse Document Frequency) scores words based on:
            - How often they appear in THIS article (TF)
            - How rare they are across ALL articles (IDF)
            
            Words like "the" score low; distinctive words score high.
            """)
        
        with st.expander("What is Passive-Aggressive Classifier?"):
            st.write("""
            A fast online linear classifier that:
            - Does nothing when predictions are correct
            - Minimally updates weights when predictions are wrong
            
            It's ideal for text classification and supports online learning.
            """)
        
        with st.expander("Why did accuracy drop to 57.9% on another dataset?"):
            st.write("""
            **Domain shift:** The model trained on Reuters articles learned:
            - "Reuters writing style = real news"
            - Words like "spokesman", "statement", weekdays = real
            
            When tested on different sources (NYT, CNN), it misclassified them as fake.
            This is a known weakness in text-based fake news detection.
            """)
        
        with st.expander("Can your model be fooled?"):
            st.write("""
            Yes. A fake article written in Reuters/journalist style would likely pass.
            
            This is why the model should be used as a **screening tool only**,
            combined with human fact-checking and professional verification.
            """)
        
        with st.expander("What should I do with the results?"):
            st.write("""
            1. Use this as a **first screening tool** to flag suspicious articles
            2. Do NOT rely on it as the final authority
            3. Cross-check with professional fact-checkers (Snopes, PolitiFact, etc.)
            4. Verify claims against multiple reliable sources
            5. Check the article's author, publication, and date
            """)
        
        with st.expander("How accurate is this really?"):
            st.write("""
            - **On familiar data:** 98.38% accuracy (excellent)
            - **On unfamiliar sources:** 57.87% accuracy (guessing)
            - **Mixed training:** 88.18% accuracy
            
            The true accuracy depends on how similar new articles are to training data.
            """)

if __name__ == "__main__":
    main()
