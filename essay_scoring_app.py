# essay_scoring_app.py

import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Essay Scoring System",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("📝 AI Essay Scoring System")
st.markdown("---")

# Load model and tokenizer (cached for performance)
@st.cache_resource
def load_model():
    """Load the fine-tuned BERT model and tokenizer"""
    model_path = r"D:\NLU\project\essay_model_fp16"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure the model path is correct and the model files exist.")
        return None, None, None

# Text cleaning function (same as training)
def clean_text(text):
    """Clean and preprocess the input text"""
    text = str(text).lower()
    # Keep only alphanumeric and spaces for English BERT
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Custom rounding function (same as training)
def custom_round(score):
    """
    Custom rounding rules:
    - If score < 1.5: round to 1
    - If 1.5 <= score < 2.5: round to 2
    - If 2.5 <= score < 3.5: round to 3
    - If 3.5 <= score < 4.5: round to 4
    - If 4.5 <= score < 5.5: round to 5
    - If score >= 5.5: round to 6
    """
    if score < 1.5:
        return 1
    elif score < 2.5:
        return 2
    elif score < 3.5:
        return 3
    elif score < 4.5:
        return 4
    elif score < 5.5:
        return 5
    else:
        return 6

# Prediction function with stride for long texts
def predict_score(text, model, tokenizer, device, max_length=512, stride=256):
    """Predict the score for a given essay, handling long texts"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize with stride (same as training)
    tokenized = tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_overflowing_tokens=True,
        stride=stride,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # Make predictions for all chunks
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        scores = outputs.logits.squeeze(-1)  # Shape: (num_chunks,)
    
    # Take mean score if multiple chunks
    if len(scores.shape) > 0 and scores.shape[0] > 1:
        raw_score = scores.mean().item()
        num_chunks = scores.shape[0]
    else:
        raw_score = scores.item() if len(scores.shape) > 0 else scores.item()
        num_chunks = 1
    
    # Apply custom rounding
    rounded_score = custom_round(raw_score)
    
    # Ensure score is within 1-6 range
    rounded_score = max(1, min(6, rounded_score))
    
    return rounded_score, raw_score, num_chunks

# Function to get score interpretation
def get_score_interpretation(score):
    """Get detailed interpretation for each score"""
    interpretations = {
        6: {
            "level": "Excellent",
            "emoji": "🌟",
            "color": "#0066cc",
            "description": "Outstanding essay with strong arguments, excellent organization, and sophisticated writing style.",
            "feedback": [
                "Exceptional clarity and coherence",
                "Advanced vocabulary and sentence structure",
                "Strong critical thinking and analysis",
                "Excellent use of evidence and examples"
            ]
        },
        5: {
            "level": "Very Good",
            "emoji": "👍",
            "color": "#00cc44",
            "description": "Well-written essay with good structure, clear arguments, and effective use of language.",
            "feedback": [
                "Clear organization and logical flow",
                "Good use of supporting details",
                "Appropriate vocabulary and grammar",
                "Minor improvements possible"
            ]
        },
        4: {
            "level": "Good",
            "emoji": "📝",
            "color": "#88cc00",
            "description": "Solid essay that demonstrates good understanding. Some areas could be developed further.",
            "feedback": [
                "Adequate structure and organization",
                "Some supporting evidence present",
                "Generally clear but could be more detailed",
                "Room for vocabulary enhancement"
            ]
        },
        3: {
            "level": "Satisfactory",
            "emoji": "📋",
            "color": "#ffaa00",
            "description": "Satisfactory essay that meets basic requirements but needs stronger development and clearer organization.",
            "feedback": [
                "Basic structure present",
                "Limited supporting details",
                "Ideas need better development",
                "Grammar and vocabulary need improvement"
            ]
        },
        2: {
            "level": "Needs Improvement",
            "emoji": "⚠️",
            "color": "#ff6644",
            "description": "Essay needs significant improvement in structure, argumentation, and clarity.",
            "feedback": [
                "Weak organization and structure",
                "Insufficient evidence and support",
                "Unclear or underdeveloped ideas",
                "Multiple grammar and vocabulary issues"
            ]
        },
        1: {
            "level": "Poor",
            "emoji": "❌",
            "color": "#ff4444",
            "description": "Essay requires major revision. Focus on developing clear arguments and improving writing mechanics.",
            "feedback": [
                "Lacks clear structure",
                "Minimal or no supporting evidence",
                "Ideas are unclear or off-topic",
                "Significant language problems"
            ]
        }
    }
    return interpretations.get(score, interpretations[1])

# Function to create score gauge
def create_score_gauge(score, min_score=1, max_score=6):
    """Create a visual gauge for the score"""
    fig, ax = plt.subplots(figsize=(5, 2), subplot_kw={'projection': 'polar'})
    
    # Normalize score - score 1 is left, score 6 is right
    normalized_score = (score - min_score) / (max_score - min_score)
    angle = np.pi * normalized_score
    
    # Create gauge background
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    # Background arc (gray)
    ax.plot(theta, r, color='#e0e0e0', linewidth=15, solid_capstyle='round')
    
    # Get interpretation for color
    interpretation = get_score_interpretation(score)
    color = interpretation['color']
    
    # Colored progress arc
    if angle > 0:
        progress_theta = np.linspace(0, angle, max(2, int(angle / np.pi * 100)))
        progress_r = np.ones_like(progress_theta)
        ax.plot(progress_theta, progress_r, color=color, linewidth=15, solid_capstyle='round')
    
    # Add needle
    ax.plot([angle, angle], [0, 1.3], color='black', linewidth=3)
    
    # Add score and level text
    ax.text(np.pi/2, 0.35, f'{score}', fontsize=30, fontweight='bold', ha='center', va='center')
    ax.text(np.pi/2, 0.1, f'{interpretation["level"]}', fontsize=10, ha='center', va='center', color=color)
    
    # Add min and max labels
    ax.text(0, 0.6, f'{min_score}', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(np.pi, 0.6, f'{max_score}', fontsize=12, fontweight='bold', ha='center', va='center')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.4)
    ax.set_frame_on(False)
    
    return fig

# Load the model
with st.spinner("🔄 Loading model... This may take a moment."):
    model, tokenizer, device = load_model()

# Main app layout
if model is not None and tokenizer is not None:
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI-powered essay scoring system uses a fine-tuned BERT model 
        to evaluate and score essays on a scale of 1-6.
        
        **Features:**
        - Handles long essays (auto-chunking)
        - Custom rounding for accurate scoring
        - Detailed feedback and analysis
        """)
        
        st.markdown("---")
        st.markdown("### 📋 How to use")
        st.markdown("""
        1. Paste or type your essay
        2. Maximum 10,000 characters
        3. Click "Score Essay"
        4. View your score and feedback
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Scoring Scale")
        scores_info = {
            "🌟 6": "Excellent",
            "👍 5": "Very Good",
            "📝 4": "Good",
            "📋 3": "Satisfactory",
            "⚠️ 2": "Needs Improvement",
            "❌ 1": "Poor"
        }
        for score, label in scores_info.items():
            st.markdown(f"{score}: {label}")
        
        st.markdown("---")
        st.caption(f"Running on: {device}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Enter Your Essay")
        
        # Text input area
        essay_text = st.text_area(
            "Paste or type your essay here:",
            height=350,
            placeholder="Type or paste your essay here...\n\nExample: Climate change is one of the most pressing issues...",
            key="essay_input"
        )
        
        # Character and word counter
        char_count = len(essay_text)
        word_count = len(essay_text.split())
        max_chars = 10000
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Characters", f"{char_count:,}", delta=f"{max_chars - char_count:,} remaining")
        with col_info2:
            st.metric("Words", f"{word_count:,}")
        with col_info3:
            est_time = "Fast" if char_count < 2000 else "Medium" if char_count < 5000 else "Long"
            st.metric("Est. Time", est_time)
        
        # Progress bar
        if char_count > max_chars:
            progress = 1.0
            st.progress(progress, text="⚠️ Character limit exceeded!")
            st.error(f"⚠️ Your essay exceeds the maximum limit! Current: {char_count:,} / {max_chars:,} max")
        else:
            progress = char_count / max_chars
            st.progress(progress)
            
            if char_count > 0 and char_count < 500:
                st.warning("📝 Essay is quite short. Consider writing more for better evaluation.")
        
        # Submit button
        submit_button = st.button(
            "🎯 Score My Essay",
            type="primary",
            disabled=(char_count == 0 or char_count > max_chars),
            use_container_width=True
        )
    
    with col2:
        st.subheader("📈 Score Result")
        
        if submit_button and essay_text and char_count <= max_chars and char_count > 0:
            with st.spinner("🔍 Analyzing your essay..."):
                try:
                    # Get prediction
                    final_score, raw_score, num_chunks = predict_score(essay_text, model, tokenizer, device)
                    
                    # Display gauge
                    fig = create_score_gauge(final_score, min_score=1, max_score=6)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Get interpretation
                    interpretation = get_score_interpretation(final_score)
                    
                    # Display score with emoji
                    st.markdown(f"### {interpretation['emoji']} Score: {final_score} - {interpretation['level']}")
                    
                    # Detailed Analysis in expander
                    with st.expander("📊 View Detailed Analysis"):
                        tab1, tab2, tab3 = st.tabs(["Scores", "Feedback", "Info"])
                        
                        with tab1:
                            st.metric("Raw Score", f"{raw_score:.3f}")
                            st.metric("Rounded Score", final_score)
                            if num_chunks > 1:
                                st.caption(f"Essay was processed in {num_chunks} chunks (long text)")
                            else:
                                st.caption("Essay fit in single chunk")
                        
                        with tab2:
                            st.markdown("### What this score means:")
                            st.info(interpretation['description'])
                            
                            st.markdown("### Specific Feedback:")
                            for feedback in interpretation['feedback']:
                                if final_score >= 4:
                                    st.success(f"✅ {feedback}")
                                elif final_score >= 2:
                                    st.warning(f"⚠️ {feedback}")
                                else:
                                    st.error(f"❌ {feedback}")
                        
                        with tab3:
                            st.markdown(f"**Essay Length:** {char_count:,} characters")
                            st.markdown(f"**Word Count:** {word_count:,} words")
                            st.markdown(f"**Chunks Processed:** {num_chunks}")
                            st.markdown(f"**Processing Device:** {device}")
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.info("Please try again with a different text.")
        else:
            # Placeholder when no score yet
            st.info("👈 Enter your essay and click 'Score My Essay' to get your score")
            
            # Show example gauge
            fig = create_score_gauge(0, min_score=1, max_score=6)
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown("### Score Interpretation")
            for s, interp in [
                (6, "Excellent essay"),
                (5, "Very good essay"),
                (4, "Good essay"),
                (3, "Satisfactory essay"),
                (2, "Needs improvement"),
                (1, "Poor essay")
            ]:
                st.markdown(f"- **{s}**: {interp}")
            
            st.caption("Your score will appear here after analysis")
    
    # Footer
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    with col_footer1:
        st.markdown("**Score Range:** 1-6")
    with col_footer2:
        st.markdown("**Max Length:** 10,000 characters")
    with col_footer3:
        st.markdown("**Model:** BERT Fine-tuned")
    
    st.markdown(
        "<div style='text-align: center; color: gray; margin-top: 20px;'>"
        "Made with ❤️ using Streamlit | Essay Scoring System v2.0"
        "</div>",
        unsafe_allow_html=True
    )
    
else:
    st.error("❌ Failed to load the model.")
    st.info("Please check the model path and ensure all files are present.")
    st.code("""
    Required files in model directory:
    - config.json
    - pytorch_model.bin (or model.safetensors)
    - tokenizer.json
    - vocab.txt
    - special_tokens_map.json
    """)