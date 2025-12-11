# ============================================================================
# COMPLETE RAG SYSTEM - STREAMLIT APP
# Save as: app.py
# ============================================================================

import streamlit as st
import pickle
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

st.set_page_config(
    page_title="RAG Movie Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .positive-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .negative-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .retrieved-review {
        background-color: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 3px solid #007bff;
    }
    .answer-box {
        background-color: #e7f3ff;
        border-left: 5px solid #007bff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CLASSES
# ============================================================================

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words -= {'not', 'no', 'nor', 'neither', 'never'}
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
        return ' '.join(text.split())
    
    def lemmatize_text(self, text):
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words or word in ['not', 'no', 'never']]
        return ' '.join(words)
    
    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.lemmatize_text(text)
        return text

# ============================================================================
# RAG SYSTEM
# ============================================================================

class RAGSystem:
    """Complete RAG system with retrieval and generation"""
    
    def __init__(self, index, review_db, classifier_data, encoder):
        self.index = index
        self.reviews = review_db['reviews']
        self.sentiments = review_db['sentiments']
        self.ratings = review_db['ratings']
        self.encoder = encoder
        
        # Classifier components
        self.classifier = classifier_data['model']
        self.tfidf = classifier_data['tfidf']
        self.label_encoder = classifier_data['label_encoder']
        
        self.preprocessor = TextPreprocessor()
        self.vader = SentimentIntensityAnalyzer()
    
    def retrieve(self, query, top_k=5):
        """Retrieve relevant reviews for a query"""
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get retrieved reviews
        retrieved = []
        for idx, dist in zip(indices[0], distances[0]):
            retrieved.append({
                'review': self.reviews[idx],
                'sentiment': self.sentiments[idx],
                'rating': self.ratings[idx] if self.ratings[idx] else 'N/A',
                'distance': float(dist),
                'relevance': 1 / (1 + dist)  # Convert distance to relevance score
            })
        
        return retrieved
    
    def generate_answer(self, query, retrieved_reviews):
        """Generate answer based on retrieved reviews"""
        
        # Count sentiments
        pos_count = sum(1 for r in retrieved_reviews if r['sentiment'] == 'positive')
        neg_count = len(retrieved_reviews) - pos_count
        
        # Calculate overall sentiment
        if pos_count > neg_count:
            overall = "positive"
            confidence = pos_count / len(retrieved_reviews)
        else:
            overall = "negative"
            confidence = neg_count / len(retrieved_reviews)
        
        # Extract key themes
        all_text = ' '.join([r['review'] for r in retrieved_reviews[:3]])
        
        # Generate natural language answer
        answer = f"Based on {len(retrieved_reviews)} relevant reviews, the overall sentiment is **{overall}** "
        answer += f"({pos_count} positive, {neg_count} negative). "
        
        # Add specific insights
        if pos_count > 0:
            pos_reviews = [r for r in retrieved_reviews if r['sentiment'] == 'positive']
            answer += f"\n\n**Positive aspects mentioned:** Viewers appreciated "
            # Extract positive keywords
            pos_words = []
            for review in pos_reviews[:2]:
                words = review['review'].lower().split()
                pos_words.extend([w for w in words if w in ['great', 'amazing', 'excellent', 'love', 'perfect', 'fantastic']])
            if pos_words:
                unique_words = list(set(pos_words))[:3]
                answer += ', '.join(unique_words) + ". "
        
        if neg_count > 0:
            neg_reviews = [r for r in retrieved_reviews if r['sentiment'] == 'negative']
            answer += f"\n\n**Negative aspects mentioned:** Critics noted "
            # Extract negative keywords
            neg_words = []
            for review in neg_reviews[:2]:
                words = review['review'].lower().split()
                neg_words.extend([w for w in words if w in ['bad', 'terrible', 'awful', 'boring', 'waste', 'poor']])
            if neg_words:
                unique_words = list(set(neg_words))[:3]
                answer += ', '.join(unique_words) + ". "
        
        return answer, overall, confidence
    
    def predict_sentiment(self, text):
        """Classify sentiment of a single review"""
        preprocessed = self.preprocessor.preprocess(text)
        X = self.tfidf.transform([preprocessed])
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = probabilities[prediction]
        sentiment = self.label_encoder.classes_[prediction]
        return sentiment, confidence
    
    def query_rag(self, query, top_k=5):
        """Complete RAG pipeline: retrieve + generate"""
        retrieved = self.retrieve(query, top_k)
        answer, sentiment, confidence = self.generate_answer(query, retrieved)
        return answer, sentiment, confidence, retrieved

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_rag_system():
    """Load complete RAG system"""
    try:
        # Load FAISS index
        index = faiss.read_index('faiss_index.bin')
        
        # Load review database
        with open('review_database.pkl', 'rb') as f:
            review_db = pickle.load(f)
        
        # Load classifier
        with open('sentiment_classifier.pkl', 'rb') as f:
            classifier_data = pickle.load(f)
        
        # Load encoder
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create RAG system
        rag = RAGSystem(index, review_db, classifier_data, encoder)
        
        return rag
    
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e.filename}")
        st.info("Required files: faiss_index.bin, review_database.pkl, sentiment_classifier.pkl")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {e}")
        st.stop()

# Load RAG system
with st.spinner("🔄 Loading RAG system..."):
    rag_system = load_rag_system()
    st.success("✅ RAG system loaded!")

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🎬 RAG Movie Review Analyzer")
st.markdown("### Ask questions about movies and get AI-generated answers with evidence")

# Tabs for different modes
tab1, tab2 = st.tabs(["🔍 Query Mode (RAG)", "📝 Review Classification"])

# ============================================================================
# TAB 1: RAG QUERY MODE
# ============================================================================

with tab1:
    st.subheader("🤖 Ask Questions About Movies")
    st.markdown("*The system will search through thousands of reviews to answer your question*")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - What do people think about the acting in this movie?
        - Are the special effects good?
        - Is this movie worth watching?
        - What are the main criticisms?
        - How's the plot and storyline?
        - What do viewers say about the cinematography?
        """)
    
    query = st.text_input(
        "Enter your question:",
        placeholder="What do people think about the acting?",
        key="query_input"
    )
    
    top_k = st.slider("Number of reviews to retrieve:", 3, 10, 5)
    
    if st.button("🔍 Search & Generate Answer", type="primary"):
        if query:
            with st.spinner("🤖 Searching reviews and generating answer..."):
                answer, sentiment, confidence, retrieved = rag_system.query_rag(query, top_k)
                
                # Display generated answer
                st.markdown("---")
                st.subheader("💡 AI-Generated Answer")
                
                st.markdown(f"""
                    <div class="answer-box">
                        <h3>📖 Answer:</h3>
                        <p style="font-size: 1.1rem; line-height: 1.6;">{answer}</p>
                        <br>
                        <p><strong>Overall Sentiment:</strong> {sentiment.upper()} 
                        <strong>Confidence:</strong> {confidence*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Answer Confidence"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#28a745" if sentiment == 'positive' else "#dc3545"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "darkgray"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display retrieved reviews (evidence)
                st.markdown("---")
                st.subheader("📚 Retrieved Reviews (Evidence)")
                st.markdown(f"*Top {len(retrieved)} most relevant reviews that support this answer:*")
                
                for i, review_data in enumerate(retrieved, 1):
                    sentiment_color = "🟢" if review_data['sentiment'] == 'positive' else "🔴"
                    relevance_pct = review_data['relevance'] * 100
                    
                    with st.expander(f"{sentiment_color} Review {i} - {review_data['sentiment'].upper()} (Relevance: {relevance_pct:.1f}%)"):
                        st.markdown(f"**Rating:** {review_data['rating']}")
                        st.markdown(f"**Sentiment:** {review_data['sentiment']}")
                        st.markdown(f"**Relevance Score:** {relevance_pct:.1f}%")
                        st.markdown("---")
                        st.write(review_data['review'][:500] + "..." if len(review_data['review']) > 500 else review_data['review'])
        else:
            st.warning("⚠️ Please enter a question")

# ============================================================================
# TAB 2: REVIEW CLASSIFICATION MODE
# ============================================================================

with tab2:
    st.subheader("📝 Classify a Single Review")
    st.markdown("*Enter a movie review to predict its sentiment*")
    
    review_text = st.text_area(
        "Enter movie review:",
        height=150,
        placeholder="This movie was absolutely amazing! The acting was superb...",
        key="review_input"
    )
    
    if st.button("🔍 Analyze Review", type="primary", key="classify_btn"):
        if review_text and len(review_text.strip()) > 10:
            with st.spinner("🤖 Analyzing..."):
                sentiment, confidence = rag_system.predict_sentiment(review_text)
                
                st.markdown("---")
                st.subheader("📊 Sentiment Prediction")
                
                if sentiment == 'positive':
                    st.markdown(f"""
                        <div class="positive-box">
                            <h2 style="color: #28a745; margin: 0;">✅ POSITIVE REVIEW</h2>
                            <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                                Confidence: <strong>{confidence*100:.1f}%</strong>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="negative-box">
                            <h2 style="color: #dc3545; margin: 0;">❌ NEGATIVE REVIEW</h2>
                            <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                                Confidence: <strong>{confidence*100:.1f}%</strong>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Find similar reviews using RAG
                st.markdown("---")
                st.subheader("🔍 Similar Reviews Found")
                
                similar = rag_system.retrieve(review_text, top_k=3)
                for i, sim in enumerate(similar, 1):
                    sentiment_icon = "🟢" if sim['sentiment'] == 'positive' else "🔴"
                    with st.expander(f"{sentiment_icon} Similar Review {i} - {sim['sentiment'].upper()}"):
                        st.write(sim['review'][:300] + "..." if len(sim['review']) > 300 else sim['review'])
        else:
            st.warning("⚠️ Please enter a review (at least 10 characters)")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("ℹ️ About RAG System")
    
    st.markdown("""
    ### 🎯 What is RAG?
    
    **RAG (Retrieval-Augmented Generation)** combines:
    
    1. **Retriever:** Searches database for relevant reviews
    2. **Generator:** Creates natural language answers
    
    ### 🔧 How It Works:
    
    1. You ask a question
    2. System searches thousands of reviews
    3. Retrieves most relevant reviews
    4. Generates answer based on evidence
    5. Shows you the source reviews
    
    ### 📊 Technology:
    
    - **Encoder:** Sentence-BERT
    - **Vector DB:** FAISS
    - **Classifier:** Logistic Regression
    - **Reviews:** {len(rag_system.reviews):,}
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Stats")
    st.metric("Total Reviews", f"{len(rag_system.reviews):,}")
    
    pos_count = sum(1 for s in rag_system.sentiments if s == 'positive')
    neg_count = len(rag_system.sentiments) - pos_count
    st.metric("Positive Reviews", f"{pos_count:,}")
    st.metric("Negative Reviews", f"{neg_count:,}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 1rem 0;">
        <p><strong>🎬 RAG-Powered Movie Review Analysis System</strong></p>
        <p>Retrieval-Augmented Generation | Sentence-BERT | FAISS | ML Classification</p>
    </div>
""", unsafe_allow_html=True)
