"""
RAG-based Movie Sentiment Analysis with Evidence Retrieval
Streamlit Application - Works with Auto-Detected Sentiment
Production Ready - No API Keys Required
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Movie Sentiment RAG System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .answer-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .positive-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    .negative-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    .neutral-badge {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_all_resources():
    """Load encoder, generator, FAISS index, and metadata"""
    
    try:
        # Load sentence transformer
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load FLAN-T5 for generation
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = generator.to(device)
        
        # Load FAISS index - FIXED PATH
        if not os.path.exists("models/faiss_index.bin"):
            return None, None, None, None, None, None, "faiss_missing"
        
        index = faiss.read_index("models/faiss_index.bin")
        
        # Load metadata - FIXED PATH
        if not os.path.exists("models/review_metadata.pkl"):
            return None, None, None, None, None, None, "metadata_missing"
        
        with open('models/review_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load build report if exists
        build_report = None
        if os.path.exists('models/build_report.json'):
            with open('models/build_report.json', 'r') as f:
                build_report = json.load(f)
        
        return encoder, tokenizer, generator, device, index, metadata, build_report
    
    except Exception as e:
        return None, None, None, None, None, None, str(e)

# ============================================================================
# CORE RAG FUNCTIONS
# ============================================================================

def retrieve_reviews(query, encoder, index, metadata, top_k=5, filter_sentiment=None):
    """Retrieve relevant reviews from FAISS index"""
    
    # Encode query
    query_embedding = encoder.encode([query], convert_to_numpy=True)
    
    # Search FAISS
    search_k = min(top_k * 3 if filter_sentiment else top_k, len(metadata['review_ids']))
    distances, indices = index.search(query_embedding.astype('float32'), search_k)
    
    # Build results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx >= len(metadata['sentiments']):
            continue
            
        sentiment = metadata['sentiments'][idx]
        
        # Apply sentiment filter
        if filter_sentiment and sentiment != filter_sentiment:
            continue
        
        result = {
            'review_id': metadata['review_ids'][idx],
            'text': metadata['texts'][idx],
            'sentiment': sentiment,
            'similarity_score': float(1 / (1 + distance))
        }
        
        if metadata['ratings'] and idx < len(metadata['ratings']):
            result['rating'] = metadata['ratings'][idx]
        
        results.append(result)
        
        if len(results) >= top_k:
            break
    
    return results

def analyze_sentiment_distribution(results):
    """Analyze sentiment in retrieved results"""
    if not results:
        return {}
    
    sentiments = [r['sentiment'] for r in results]
    pos_count = sentiments.count('positive')
    neg_count = sentiments.count('negative')
    neutral_count = sentiments.count('neutral')
    total = len(sentiments)
    
    return {
        'positive_count': pos_count,
        'negative_count': neg_count,
        'neutral_count': neutral_count,
        'positive_pct': (pos_count / total * 100) if total > 0 else 0,
        'negative_pct': (neg_count / total * 100) if total > 0 else 0,
        'neutral_pct': (neutral_count / total * 100) if total > 0 else 0,
        'dominant': 'positive' if pos_count > neg_count else ('negative' if neg_count > pos_count else 'neutral')
    }

def generate_explanation(query, results, tokenizer, generator, device):
    """Generate natural language explanation with citations"""
    
    # Build context from top results
    context_parts = []
    for i, result in enumerate(results[:4], 1):
        snippet = result['text'][:400]
        rating = f", Rating: {result['rating']}/10" if 'rating' in result else ""
        context_parts.append(
            f"Review #{result['review_id']} ({result['sentiment'].upper()}{rating}):\n{snippet}"
        )
    
    context = "\n\n".join(context_parts)
    sentiment_dist = analyze_sentiment_distribution(results)
    
    # Create prompt
    prompt = f"""Analyze movie reviews and answer this question: {query}

Evidence from reviews:
{context}

Sentiment: {sentiment_dist['positive_count']} positive, {sentiment_dist['negative_count']} negative

Provide a clear answer that:
1. Directly answers the question
2. References specific Review IDs (e.g., "Review #123 mentions...")
3. Explains the sentiment reasoning

Answer:"""

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = generator.generate(
            **inputs,
            max_length=200,
            num_beams=4,
            temperature=0.7,
            do_sample=False,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🎬 Movie Sentiment RAG System</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        <p>Analyze movie review sentiment with AI-powered evidence retrieval and explanations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    with st.spinner("🔄 Loading AI models and review database..."):
        encoder, tokenizer, generator, device, index, metadata, error_or_report = load_all_resources()
        
        # Handle errors
        if encoder is None:
            if error_or_report == "faiss_missing":
                st.error("""
                ❌ **Missing FAISS Index**
                
                The file `models/faiss_index.bin` was not found.
                
                **Solution:**
                1. Run `python build_index.py` locally
                2. Commit `models/faiss_index.bin` to GitHub
                3. Redeploy on Streamlit Cloud
                
                **Note:** Make sure you have `data/imdb_sup.csv` with your reviews before building the index.
                """)
            elif error_or_report == "metadata_missing":
                st.error("""
                ❌ **Missing Metadata**
                
                The file `models/review_metadata.pkl` was not found.
                
                **Solution:**
                1. Run `python build_index.py` locally
                2. Commit `models/review_metadata.pkl` to GitHub
                3. Redeploy on Streamlit Cloud
                """)
            else:
                st.error(f"""
                ❌ **Error Loading Resources**
                
                {error_or_report}
                
                **Common causes:**
                - Missing model files
                - Corrupted index files
                - Insufficient memory on Streamlit Cloud
                
                **Solution:** Check deployment logs and ensure all files are properly committed.
                """)
            st.stop()
        
        total_reviews = len(metadata['review_ids'])
        build_report = error_or_report if isinstance(error_or_report, dict) else None
    
    # Show build info if available
    if build_report:
        with st.expander("ℹ️ Dataset Information", expanded=False):
            st.markdown(f"""
            <div class="info-box">
                <strong>Index Build Details:</strong><br>
                📊 Total Reviews: {build_report.get('total_reviews', 'N/A'):,}<br>
                ✅ Positive: {build_report.get('positive_reviews', 'N/A'):,} ({build_report.get('positive_percentage', 0):.1f}%)<br>
                ❌ Negative: {build_report.get('negative_reviews', 'N/A'):,} ({build_report.get('negative_percentage', 0):.1f}%)<br>
                🔍 Sentiment Source: {build_report.get('sentiment_source', 'Unknown').replace('_', ' ').title()}<br>
                💾 Index Size: {build_report.get('index_size_mb', 0):.2f} MB
            </div>
            """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        top_k = st.slider(
            "Reviews to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of similar reviews to find"
        )
        
        # Get available sentiments from metadata
        unique_sentiments = set(metadata['sentiments'])
        sentiment_options = ["All"] + sorted([s.title() for s in unique_sentiments if s])
        
        filter_sentiment = st.selectbox(
            "Filter by sentiment",
            sentiment_options
        )
        filter_val = None if filter_sentiment == "All" else filter_sentiment.lower()
        
        show_sources = st.checkbox("Show source reviews", value=True)
        show_chart = st.checkbox("Show sentiment chart", value=True)
        
        st.divider()
        st.header("📊 Database Info")
        st.metric("Total Reviews", f"{total_reviews:,}")
        
        # Count sentiments
        sentiment_counts = {}
        for s in metadata['sentiments']:
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
        
        for sentiment, count in sorted(sentiment_counts.items()):
            if sentiment:
                st.metric(sentiment.title(), f"{count:,}")
        
        st.caption(f"Running on: {device.upper()}")
        
        st.divider()
        st.header("💡 Try These")
        examples = [
            "Why do some movies get negative reviews?",
            "What makes a horror movie scary?",
            "Why do viewers love good acting?",
            "What are common complaints about bad movies?",
            "What makes cinematography great?",
            "Why do some comedies fail?",
            "What creates emotional impact in films?"
        ]
        
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.example_query = ex
    
    # Main interface
    st.header("🔍 Ask About Movie Reviews")
    
    default_query = st.session_state.get('example_query', '')
    if 'example_query' in st.session_state:
        del st.session_state.example_query
    
    query = st.text_input(
        "Enter your question:",
        value=default_query,
        placeholder="e.g., Why do some movies receive negative sentiment?",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("🔎 Analyze", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    # Process query
    if analyze_btn and query:
        with st.spinner("🤔 Analyzing reviews and generating explanation..."):
            try:
                start = datetime.now()
                
                # Retrieve
                results = retrieve_reviews(
                    query, encoder, index, metadata,
                    top_k=top_k,
                    filter_sentiment=filter_val
                )
                
                if not results:
                    st.warning("⚠️ No reviews found. Try different filters or query.")
                    st.stop()
                
                # Analyze sentiment
                sent_dist = analyze_sentiment_distribution(results)
                
                # Generate explanation
                answer = generate_explanation(query, results, tokenizer, generator, device)
                
                elapsed = (datetime.now() - start).total_seconds()
                
                # Display results
                st.divider()
                st.subheader("📝 Analysis Results")
                
                # Metrics
                cols = st.columns(5)
                cols[0].metric("Sources", len(results))
                cols[1].metric("Positive", sent_dist['positive_count'])
                cols[2].metric("Negative", sent_dist['negative_count'])
                if sent_dist.get('neutral_count', 0) > 0:
                    cols[3].metric("Neutral", sent_dist['neutral_count'])
                cols[4].metric("Time", f"{elapsed:.1f}s")
                
                # Answer
                st.subheader("💡 AI Explanation")
                st.markdown(f"""
                <div class="answer-box">
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Chart
                if show_chart:
                    st.subheader("📊 Sentiment Distribution")
                    
                    chart_data = []
                    chart_colors = []
                    chart_labels = []
                    
                    if sent_dist['positive_count'] > 0:
                        chart_data.append(sent_dist['positive_count'])
                        chart_colors.append('#28a745')
                        chart_labels.append('Positive')
                    
                    if sent_dist['negative_count'] > 0:
                        chart_data.append(sent_dist['negative_count'])
                        chart_colors.append('#dc3545')
                        chart_labels.append('Negative')
                    
                    if sent_dist.get('neutral_count', 0) > 0:
                        chart_data.append(sent_dist['neutral_count'])
                        chart_colors.append('#ffc107')
                        chart_labels.append('Neutral')
                    
                    if chart_data:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=chart_labels,
                                y=chart_data,
                                marker_color=chart_colors,
                                text=chart_data,
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=20, b=20),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Sources
                if show_sources:
                    st.subheader("📚 Source Reviews (Evidence)")
                    for i, result in enumerate(results[:5], 1):
                        sentiment = result['sentiment']
                        if sentiment == 'positive':
                            badge = "positive-badge"
                        elif sentiment == 'negative':
                            badge = "negative-badge"
                        else:
                            badge = "neutral-badge"
                        
                        rating_txt = f" | ⭐ {result['rating']}/10" if 'rating' in result else ""
                        
                        with st.expander(
                            f"Review #{result['review_id']} - {sentiment.title()} "
                            f"(Relevance: {result['similarity_score']:.2f}){rating_txt}",
                            expanded=(i <= 2)
                        ):
                            st.markdown(f'<span class="{badge}">{sentiment.upper()}</span>', 
                                      unsafe_allow_html=True)
                            st.caption(f"Similarity: {result['similarity_score']:.3f}")
                            if 'rating' in result:
                                st.caption(f"Rating: ⭐ {result['rating']}/10")
                            st.write("**Review Text:**")
                            st.write(result['text'][:600] + ("..." if len(result['text']) > 600 else ""))
                
                # Export
                st.divider()
                st.subheader("💾 Export Results")
                
                export_data = {
                    'query': query,
                    'answer': answer,
                    'sentiment_distribution': sent_dist,
                    'processing_time_seconds': elapsed,
                    'sources': results[:5]
                }
                
                col1, col2 = st.columns(2)
                col1.download_button(
                    "📥 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                sources_df = pd.DataFrame([
                    {
                        'Review_ID': r['review_id'],
                        'Sentiment': r['sentiment'],
                        'Relevance': r['similarity_score'],
                        'Rating': r.get('rating', 'N/A'),
                        'Text': r['text'][:200]
                    }
                    for r in results[:5]
                ])
                col2.download_button(
                    "📥 Download CSV",
                    data=sources_df.to_csv(index=False),
                    file_name=f"sources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                with st.expander("🔍 Show error details"):
                    st.exception(e)
    
    elif analyze_btn:
        st.warning("⚠️ Please enter a question first!")
    
    # Footer
    st.divider()
    st.caption("🎬 RAG Sentiment Analysis System | Powered by Sentence Transformers & FLAN-T5")
    st.caption("Sentiment detection: Keyword-based analysis + existing labels")

if __name__ == "__main__":
    main()
