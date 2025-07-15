import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import io
import requests
from sentence_transformers import SentenceTransformer


# Configure page
st.set_page_config(
    page_title="Medical Vision AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .feature-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .result-box {
        background: #ecfdf5;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #10b981;
        margin-top: 1rem;
    }
    .error-box {
        background: #fef2f2;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ef4444;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_medical_vqa_model():
    """Load medical VQA model - using lighter BLIP model for better deployment"""
    try:
        model_name = "sharawy53/final_diploma_blip-med-rad-arabic"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"Error loading VQA model: {str(e)}")
        return None, None

@st.cache_resource
def load_translation_model():
    """Load Arabic-English translation model"""
    try:
      #  model_name = "facebook/nllb-200-distilled-600M"
        model_name = "google/mt5-small"
    
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading translation model: {str(e)}")
        return None, None

def analyze_medical_image(image, question, processor, model):
    """Analyze medical image with VQA"""
    try:
        # Process image and question
        inputs = processor(image, question, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=5)
        
        # Decode response
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def translate_arabic_to_english(text, tokenizer, model):
    """Translate Arabic text to English"""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        
        translated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        return f"Error translating text: {str(e)}"

def get_medical_context(question):
    """Add medical context to questions"""
    medical_keywords = {
        "xray": "X-ray medical imaging",
        "ct": "CT scan medical imaging", 
        "mri": "MRI medical imaging",
        "fracture": "bone fracture medical condition",
        "pneumonia": "lung infection medical condition",
        "tumor": "abnormal growth medical condition"
    }
    
    for keyword, context in medical_keywords.items():
        if keyword.lower() in question.lower():
            return f"In the context of {context}: {question}"
    return question

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Vision AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", 
                                   ["Medical Image Analysis", "Arabic Translation", "About"])
    
    # Add model status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Models Status")
    
    if app_mode == "Medical Image Analysis":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📊 Medical Image Analysis")
        st.write("Upload a medical image and ask questions about it using AI-powered visual question answering.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load models with status display
        with st.spinner("Loading AI models..."):
            vqa_processor, vqa_model = load_medical_vqa_model()
        
        if vqa_processor and vqa_model:
            st.sidebar.success("✅ VQA Model: Ready")
            
            # File upload
            uploaded_file = st.file_uploader("Choose a medical image...", 
                                           type=["jpg", "jpeg", "png", "bmp"],
                                           help="Supported formats: JPG, PNG, BMP")
            
            if uploaded_file is not None:
                # Display image
                image = Image.open(uploaded_file).convert("RGB")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Uploaded Medical Image", use_container_width=True)
                    st.info(f"Image size: {image.size[0]}x{image.size[1]} pixels")
                
                with col2:
                    # Question input with examples
                    st.subheader("Ask a Medical Question")
                    
                    # Quick question buttons
                    st.write("**Quick Questions:**")
                    col_q1, col_q2 = st.columns(2)
                    with col_q1:
                        if st.button("What do you see?"):
                            st.session_state.question = "What abnormalities or findings do you see in this medical image?"
                        if st.button("Any fractures?"):
                            st.session_state.question = "Are there any fractures or broken bones visible?"
                    with col_q2:
                        if st.button("Normal or abnormal?"):
                            st.session_state.question = "Does this medical image appear normal or abnormal?"
                        if st.button("Describe findings"):
                            st.session_state.question = "Describe the key medical findings in this image"
                    
                    # Custom question input
                    question = st.text_area("Or ask your own question:", 
                                           value=st.session_state.get('question', ''),
                                           placeholder="What abnormalities do you see in this X-ray?",
                                           height=100)
                    
                    if st.button("🔍 Analyze Image", type="primary"):
                        if question:
                            # Add medical context
                            contextualized_question = get_medical_context(question)
                            
                            with st.spinner("Analyzing medical image..."):
                                result = analyze_medical_image(image, contextualized_question, vqa_processor, vqa_model)
                            
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.subheader("🔍 Analysis Result:")
                            st.write(result)
                            
                            # Add confidence disclaimer
                            st.caption("⚠️ **Medical AI Disclaimer**: This analysis is for educational purposes only. Always consult healthcare professionals for medical decisions.")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Allow follow-up questions
                            if st.button("Ask Follow-up Question"):
                                st.session_state.follow_up = True
                        else:
                            st.warning("Please enter a question about the image.")
        else:
            st.sidebar.error("❌ VQA Model: Failed to load")
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error("**Model Loading Error**: The medical VQA model failed to load. This might be due to:")
            st.write("- Insufficient memory resources")
            st.write("- Network connectivity issues") 
            st.write("- Model compatibility problems")
            st.write("\n**Please try refreshing the page or contact support.**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif app_mode == "Arabic Translation":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🌐 Arabic to English Medical Translation")
        st.write("Translate Arabic medical text to English using specialized AI models.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load translation model
        with st.spinner("Loading translation model..."):
            translation_tokenizer, translation_model = load_translation_model()
        
        if translation_tokenizer and translation_model:
            st.sidebar.success("✅ Translation Model: Ready")
            
            # Example texts
            st.subheader("Example Medical Texts")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Headache"):
                    st.session_state.arabic_text = "أعاني من صداع شديد"
            with col2:
                if st.button("Chest Pain"):
                    st.session_state.arabic_text = "أشعر بألم في الصدر"
            with col3:
                if st.button("Fever"):
                    st.session_state.arabic_text = "لدي حمى وارتفاع في درجة الحرارة"
            
            arabic_text = st.text_area("Enter Arabic medical text:", 
                                     value=st.session_state.get('arabic_text', ''),
                                     placeholder="أدخل النص الطبي العربي هنا...",
                                     height=150,
                                     help="Enter Arabic text related to medical symptoms, conditions, or questions")
            
            if st.button("🔄 Translate", type="primary"):
                if arabic_text.strip():
                    with st.spinner("Translating Arabic to English..."):
                        translated_text = translate_arabic_to_english(arabic_text, 
                                                                    translation_tokenizer, 
                                                                    translation_model)
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("📝 Translation Result:")
                    st.write(f"**English:** {translated_text}")
                    st.write(f"**Arabic:** {arabic_text}")
                    
                    # Copy button simulation
                    st.text_area("Copy translated text:", value=translated_text, height=60)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("Please enter Arabic text to translate.")
        else:
            st.sidebar.error("❌ Translation Model: Failed to load")
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error("**Translation Model Error**: Failed to load the Arabic-English translation model.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif app_mode == "About":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("ℹ️ About Medical Vision AI Assistant")
        st.write("""
        This application combines advanced AI technologies to assist with medical image analysis and translation:
        
        **🔍 Features:**
        - **Medical Image Analysis**: Upload medical images (X-rays, CT scans, MRIs) and ask questions
        - **Arabic Translation**: Translate Arabic medical text to English
        - **AI-Powered**: Uses state-of-the-art vision and language models
        - **Medical Context**: Specialized for medical terminology and scenarios
        
        **🛠️ Technologies Used:**
        - **Streamlit**: Web interface framework
        - **BLIP**: Vision-language model for image question answering
        - **Helsinki-NLP**: Neural machine translation for Arabic-English
        - **PyTorch**: Deep learning framework
        - **Transformers**: Hugging Face model library
        
        **📋 Supported:**
        - **Image Types**: X-rays, CT scans, MRIs, ultrasounds
        - **Formats**: JPG, PNG, BMP
        - **Languages**: Arabic ↔ English translation
        - **Medical Domains**: Radiology, general medicine, symptoms
        
        **⚠️ Important Disclaimers:**
        - This tool is for **educational and research purposes only**
        - **NOT a substitute** for professional medical diagnosis
        - Always consult qualified healthcare professionals
        - AI responses may contain errors or limitations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System info
        st.subheader("🔧 System Information")
        try:
            import torch
            st.write(f"- PyTorch Version: {torch.__version__}")
            st.write(f"- Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            st.write(f"- Streamlit Version: {st.__version__}")
        except:
            st.write("- System information unavailable")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Medical AI Disclaimer:** This is a demonstration application for educational purposes. Always consult with qualified healthcare professionals for medical decisions, diagnosis, and treatment.")

if __name__ == "__main__":
    main()
