import streamlit as st
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests
from googletrans import Translator
import io
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #3b82f6 0%, #1e40af 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    .result-box {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vqa_model' not in st.session_state:
    st.session_state.vqa_model = None
if 'vqa_processor' not in st.session_state:
    st.session_state.vqa_processor = None
if 'translator' not in st.session_state:
    st.session_state.translator = None

@st.cache_resource
def load_medical_vqa_model():
    """Load the medical VQA model with error handling"""
    try:
        processor = BlipProcessor.from_pretrained("sharawy53/final_diploma_blip-med-rad-arabic")
        model = BlipForQuestionAnswering.from_pretrained("sharawy53/final_diploma_blip-med-rad-arabic")
        return processor, model
    except Exception as e:
        st.error(f"Error loading VQA model: {str(e)}")
        return None, None

@st.cache_resource
def load_translator():
    """Load Google Translator"""
    try:
        return Translator()
    except Exception as e:
        st.error(f"Error loading translator: {str(e)}")
        return None

def analyze_medical_image(image, question, processor, model):
    """Analyze medical image with enhanced medical context"""
    try:
        # Add medical context to the question
        medical_context = f"In a medical context, {question}"
        
        # Process the image and question
        inputs = processor(image, medical_context, return_tensors="pt")
        
        # Generate answer
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100, num_beams=5)
        
        # Decode the answer
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Enhanced medical interpretation
        if any(term in question.lower() for term in ['diagnosis', 'condition', 'disease', 'symptom']):
            disclaimer = "\n\n⚠️ Medical Disclaimer: This is an AI analysis for educational purposes only. Always consult qualified medical professionals for actual diagnosis and treatment."
            answer += disclaimer
            
        return answer
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def translate_text(text, translator, target_lang='en'):
    """Translate text using Google Translate"""
    try:
        if translator is None:
            return "Translation service unavailable"
        
        result = translator.translate(text, dest=target_lang)
        return result.text
    except Exception as e:
        return f"Translation error: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical AI Assistant</h1>
        <p>Advanced Medical Image Analysis & Arabic Translation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["Medical Image Analysis", "Arabic Translation", "About"]
    )
    
    if app_mode == "Medical Image Analysis":
        st.markdown("""
        <div class="feature-card">
            <h2>📊 Medical Image Analysis</h2>
            <p>Upload a medical image and ask questions about it using our AI-powered Visual Question Answering system.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load models
        if st.session_state.vqa_processor is None or st.session_state.vqa_model is None:
            with st.spinner("Loading medical analysis model..."):
                processor, model = load_medical_vqa_model()
                st.session_state.vqa_processor = processor
                st.session_state.vqa_model = model
        
        if st.session_state.vqa_processor is not None and st.session_state.vqa_model is not None:
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Medical Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a medical image (X-ray, MRI, CT scan, etc.)"
            )
            
            if uploaded_file is not None:
                # Display image
                image = Image.open(uploaded_file)
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Uploaded Medical Image", use_container_width=True)
                
                with col2:
                    # Question input
                    question = st.text_area(
                        "Ask a question about the image:",
                        placeholder="e.g., What abnormalities do you see in this X-ray?",
                        height=100
                    )
                    
                    # Predefined questions
                    st.subheader("Quick Questions:")
                    quick_questions = [
                        "What do you see in this medical image?",
                        "Are there any abnormalities visible?",
                        "What type of medical scan is this?",
                        "Describe the anatomical structures visible",
                        "What could be the potential diagnosis?"
                    ]
                    
                    for q in quick_questions:
                        if st.button(q, key=f"q_{hash(q)}"):
                            question = q
                    
                    # Analyze button
                    if st.button("🔍 Analyze Image", type="primary"):
                        if question:
                            with st.spinner("Analyzing medical image..."):
                                result = analyze_medical_image(
                                    image, 
                                    question, 
                                    st.session_state.vqa_processor, 
                                    st.session_state.vqa_model
                                )
                                
                                st.markdown(f"""
                                <div class="result-box">
                                    <h4>🎯 Analysis Result:</h4>
                                    <p>{result}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("Please enter a question about the image.")
        else:
            st.error("Failed to load the medical analysis model. Please refresh the page.")
    
    elif app_mode == "Arabic Translation":
        st.markdown("""
        <div class="feature-card">
            <h2>🔤 Arabic to English Translation</h2>
            <p>Translate Arabic medical texts to English using our advanced translation system.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load translator
        if st.session_state.translator is None:
            with st.spinner("Loading translation service..."):
                translator = load_translator()
                st.session_state.translator = translator
        
        if st.session_state.translator is not None:
            # Text input
            arabic_text = st.text_area(
                "Enter Arabic text:",
                placeholder="أدخل النص العربي هنا...",
                height=150,
                help="Enter Arabic text that you want to translate to English"
            )
            
            # Translation options
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔄 Translate to English", type="primary"):
                    if arabic_text.strip():
                        with st.spinner("Translating..."):
                            result = translate_text(arabic_text, st.session_state.translator)
                            
                            st.markdown(f"""
                            <div class="result-box">
                                <h4>📝 Translation Result:</h4>
                                <p><strong>Arabic:</strong> {arabic_text}</p>
                                <p><strong>English:</strong> {result}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("Please enter some Arabic text to translate.")
            
            with col2:
                if st.button("🔄 Reverse (EN→AR)"):
                    if arabic_text.strip():
                        with st.spinner("Translating..."):
                            result = translate_text(arabic_text, st.session_state.translator, target_lang='ar')
                            
                            st.markdown(f"""
                            <div class="result-box">
                                <h4>📝 Reverse Translation:</h4>
                                <p><strong>Input:</strong> {arabic_text}</p>
                                <p><strong>Arabic:</strong> {result}</p>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.error("Translation service is currently unavailable.")
    
    elif app_mode == "About":
        st.markdown("""
        <div class="feature-card">
            <h2>ℹ️ About This Application</h2>
            <p>This Medical AI Assistant combines cutting-edge computer vision and natural language processing to provide:</p>
            <ul>
                <li><strong>Medical Image Analysis:</strong> AI-powered visual question answering for medical images</li>
                <li><strong>Arabic Translation:</strong> Seamless translation between Arabic and English</li>
                <li><strong>User-Friendly Interface:</strong> Intuitive design for healthcare professionals and students</li>
            </ul>
            
            <h3>🔧 Technology Stack:</h3>
            <ul>
                <li>BLIP (Bootstrapped Language-Image Pre-training) for VQA</li>
                <li>Google Translate API for translation services</li>
                <li>Streamlit for the web interface</li>
                <li>PyTorch for deep learning operations</li>
            </ul>
            
            <h3>🎯 Use Cases:</h3>
            <ul>
                <li>Medical education and training</li>
                <li>Preliminary image analysis</li>
                <li>Medical document translation</li>
                <li>Research assistance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚠️ <strong>Disclaimer:</strong> This application is for educational and research purposes only. 
        Always consult qualified medical professionals for actual medical diagnosis and treatment.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
