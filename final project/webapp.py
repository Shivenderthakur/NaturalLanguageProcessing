import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from PIL import Image
import pytesseract
import os

# Set Tesseract path (update for your system)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Configure page
st.set_page_config(
    page_title="New Classifier",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        max-width: 800px;
        padding: 2rem;
    }
    .header {
        text-align: center;
        border-bottom: 2px solid #eee;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .developer-section {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return NewsClassifierPipeline()

class NewsClassifierPipeline:
    def __init__(self):
        self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.title_vectorizer = joblib.load('title_vectorizer.pkl')
        self.models = {
            'svm': joblib.load('svm_model.pkl'),
            'rf': joblib.load('rf_model.pkl'),
            'mlp': joblib.load('mlp_model.pkl'),
            'logistic': joblib.load('Logistic_model.pkl')
        }

    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        words = text.split()
        words = [self.lemmatizer.lemmatize(word)
                for word in words if word not in self.stop_words]
        return ' '.join(words)

    def predict(self, news_title):
        processed = self.preprocess_text(news_title)
        vectorized = self.title_vectorizer.transform([processed])
        predictions = [model.predict(vectorized)[0] for model in self.models.values()]
        real_votes = predictions.count(1)
        return 'Real' if real_votes > len(predictions)/2 else 'Fake'

def main():
    pipeline = load_pipeline()
    
    # Header
    st.markdown('<div class="header"><h1>🔍 NewsGuardian AI</h1></div>', unsafe_allow_html=True)
    
    # Input Section
    input_method = st.radio("Select Input Method:", 
                          ["Text Input", "Image/Camera Capture"],
                          horizontal=True)
    
    text = ""
    
    if input_method == "Image/Camera Capture":
        img_file = st.file_uploader("Upload or Capture Image:", 
                                  type=["png", "jpg", "jpeg"],
                                  accept_multiple_files=False,
                                  help="Upload image or use camera capture")
        if img_file:
            image = Image.open(img_file)
            st.image(image, use_column_width=True)
            text = pytesseract.image_to_string(image)
            if not text:
                st.warning("No text detected in the image")
    
    else:
        text = st.text_area("Enter News Content:", height=150,
                           placeholder="Paste news article or headline here...")
    
    # Analysis Section
    if st.button("Analyze Authenticity", use_container_width=True):
        if text.strip():
            prediction = pipeline.predict(text)
            bg_color = "#e6f3ff" if prediction == "Real" else "#ffe6e6"
            border_color = "#0068c9" if prediction == "Real" else "#c90000"
            
            st.markdown(f"""
            <div class="result-box" style="background-color: {bg_color}; border: 2px solid {border_color}">
                <h2 style="text-align: center; color: {border_color};">{prediction}</h2>
            </div>
            """, unsafe_allow_html=True)
            print(text)
            with st.expander("Detailed Analysis"):
                processed_text = pipeline.preprocess_text(text)
                st.subheader("Processed Text")
                st.caption(processed_text)
                
                st.subheader("Model Consensus")
                vectorized = pipeline.title_vectorizer.transform([processed_text])
                predictions = {name: model.predict(vectorized)[0] 
                             for name, model in pipeline.models.items()}
                for model, pred in predictions.items():
                    st.write(f"• {model.upper()}: {pred}")
        else:
            st.error("Please provide input content for analysis")

    # Developer Section
    st.markdown("""
    <div class="developer-section">
        <a href="https://in.linkedin.com/in/shivender-singh-thakur" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="40">
        </a>
        <a href="https://github.com/Shivenderthakur/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="40">
        </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()