import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import re

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
    .translation-box {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #0ea5e9;
        margin: 1rem 0;
    }
    .quick-questions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .quick-btn {
        flex: 1;
        min-width: 120px;
    }
    .language-tabs {
        display: flex;
        margin-bottom: 1rem;
    }
    .lang-tab {
        padding: 0.5rem 1rem;
        cursor: pointer;
        border-bottom: 2px solid transparent;
    }
    .lang-tab.active {
        border-bottom: 2px solid #3b82f6;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_medical_vqa_model():
    """Load medical VQA model"""
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
    """Load translation model"""
    try:
        # Using a more accurate medical translation model
        model_name = "Helsinki-NLP/opus-mt-ar-en"
        ar_en_tokenizer = AutoTokenizer.from_pretrained(model_name)
        ar_en_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # For English to Arabic
        model_name_en_ar = "Helsinki-NLP/opus-mt-en-ar"
        en_ar_tokenizer = AutoTokenizer.from_pretrained(model_name_en_ar)
        en_ar_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_en_ar)
        
        return {
            'ar_en_tokenizer': ar_en_tokenizer,
            'ar_en_model': ar_en_model,
            'en_ar_tokenizer': en_ar_tokenizer,
            'en_ar_model': en_ar_model
        }
    except Exception as e:
        st.error(f"Error loading translation model: {str(e)}")
        return None

def is_arabic(text):
    """Check if text contains Arabic characters"""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def translate_text(text, translation_models):
    """Translate text between Arabic and English"""
    if not text.strip():
        return ""
    
    try:
        if is_arabic(text):
            # Arabic to English translation
            inputs = translation_models['ar_en_tokenizer'](text, return_tensors="pt", 
                                                          padding=True, truncation=True, 
                                                          max_length=512)
            with torch.no_grad():
                generated_ids = translation_models['ar_en_model'].generate(**inputs, 
                                                                          max_length=128, 
                                                                          num_beams=4, 
                                                                          early_stopping=True)
            translated = translation_models['ar_en_tokenizer'].decode(generated_ids[0], 
                                                                     skip_special_tokens=True)
        else:
            # English to Arabic translation
            inputs = translation_models['en_ar_tokenizer'](text, return_tensors="pt", 
                                                          padding=True, truncation=True, 
                                                          max_length=512)
            with torch.no_grad():
                generated_ids = translation_models['en_ar_model'].generate(**inputs, 
                                                                          max_length=128, 
                                                                          num_beams=4, 
                                                                          early_stopping=True)
            translated = translation_models['en_ar_tokenizer'].decode(generated_ids[0], 
                                                                     skip_special_tokens=True)
        return translated
    except Exception as e:
        return f"Translation error: {str(e)}"

def analyze_medical_image(image, question, processor, model):
    """Analyze medical image with VQA"""
    try:
        # Process image and question
        inputs = processor(image, question, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            out = model.generate(**inputs, max_length=100, num_beams=5)
        
        # Decode response
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def get_medical_context(question):
    """Add medical context to questions"""
    medical_keywords = {
        "xray": "X-ray medical imaging",
        "x-ray": "X-ray medical imaging",
        "ct": "CT scan medical imaging", 
        "mri": "MRI medical imaging",
        "fracture": "bone fracture medical condition",
        "pneumonia": "lung infection medical condition",
        "tumor": "abnormal growth medical condition",
        "cancer": "cancerous growth medical condition",
        "infection": "bacterial or viral infection",
        "ultrasound": "ultrasound medical imaging",
        "scan": "medical imaging scan",
        "diagnosis": "medical diagnosis",
        "symptom": "medical symptom"
    }
    
    for keyword, context in medical_keywords.items():
        if keyword.lower() in question.lower():
            return f"In the context of {context}: {question}"
    return question

def main():
    # Initialize session state
    if 'question' not in st.session_state:
        st.session_state.question = ''
    if 'lang' not in st.session_state:
        st.session_state.lang = 'english'
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Vision AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", 
                                   ["Medical Image Analysis", "About"])
    
    # Add model status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Models Status")
    
    if app_mode == "Medical Image Analysis":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📊 Medical Image Analysis")
        st.write("Upload a medical image and ask questions about it using AI-powered visual question answering.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load models
        with st.spinner("Loading medical AI models..."):
            vqa_processor, vqa_model = load_medical_vqa_model()
            translation_models = load_translation_model()
        
        if vqa_processor and vqa_model and translation_models:
            st.sidebar.success("✅ Medical VQA Model: Ready")
            st.sidebar.success("✅ Translation Model: Ready")
            
            # File upload
            uploaded_file = st.file_uploader("Choose a medical image...", 
                                           type=["jpg", "jpeg", "png", "bmp"],
                                           help="Supported formats: JPG, PNG, BMP")
            
            if uploaded_file is not None:
                # Display image
                image = Image.open(uploaded_file).convert("RGB")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Uploaded Medical Image", use_column_width=True)
                    st.info(f"Image size: {image.size[0]}x{image.size[1]} pixels")
                
                with col2:
                    # Language tabs
                    st.markdown('<div class="language-tabs">', unsafe_allow_html=True)
                    if st.button("English", key="en_btn"):
                        st.session_state.lang = 'english'
                    if st.button("العربية", key="ar_btn"):
                        st.session_state.lang = 'arabic'
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Question input
                    st.subheader("Ask a Medical Question")
                    
                    # Suggested questions
                    st.write("**Suggested Questions:**")
                    
                    # English questions
                    en_questions = [
                        "What abnormalities do you see?",
                        "Are there any fractures?",
                        "Is this result normal or abnormal?",
                        "Describe the key findings",
                        "Any signs of infection?"
                    ]
                    
                    # Arabic questions
                    ar_questions = [
                        "ما هي التشوهات التي تراها؟",
                        "هل هناك أي كسور؟",
                        "هل هذه النتيجة طبيعية أم غير طبيعية؟",
                        "صف النتائج الرئيسية",
                        "هل هناك أي علامات للعدوى؟"
                    ]
                    
                    # Display suggested questions based on selected language
                    st.markdown('<div class="quick-questions">', unsafe_allow_html=True)
                    
                    if st.session_state.lang == 'english':
                        for q in en_questions:
                            if st.button(q, key=f"en_q_{q}", use_container_width=True, 
                                        help="Click to use this question"):
                                st.session_state.question = q
                    else:
                        for q in ar_questions:
                            if st.button(q, key=f"ar_q_{q}", use_container_width=True,
                                        help="انقر لاستخدام هذا السؤال"):
                                st.session_state.question = q
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Custom question input
                    placeholder = "What abnormalities do you see?" if st.session_state.lang == 'english' else "ما هي التشوهات التي تراها؟"
                    question = st.text_area("Your question:", 
                                           value=st.session_state.get('question', ''),
                                           placeholder=placeholder,
                                           height=100)
                    
                    if st.button("🔍 Analyze Image", type="primary"):
                        if question:
                            # Translate question to Arabic if needed (model requires Arabic)
                            original_question = question
                            
                            # If question is in English, translate to Arabic for the model
                            if not is_arabic(question):
                                with st.spinner("Translating question to Arabic..."):
                                    question = translate_text(question, translation_models)
                            
                            # Add medical context
                            contextualized_question = get_medical_context(question)
                            
                            # Analyze image
                            with st.spinner("Analyzing medical image..."):
                                arabic_answer = analyze_medical_image(image, contextualized_question, 
                                                                     vqa_processor, vqa_model)
                            
                            # Translate answer to English
                            with st.spinner("Translating answer to English..."):
                                english_answer = translate_text(arabic_answer, translation_models)
                            
                            # Display results
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.subheader("🔍 Analysis Result")
                            
                            # Display question in both languages
                            st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                            st.write(f"**Your Question (English):** {original_question if not is_arabic(original_question) else translate_text(original_question, translation_models)}")
                            st.write(f"**سؤالك (العربية):** {question if is_arabic(question) else translate_text(question, translation_models)}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Display answer in both languages
                            st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                            st.write(f"**Answer (English):** {english_answer}")
                            st.write(f"**الإجابة (العربية):** {arabic_answer}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Add confidence disclaimer
                            st.caption("⚠️ **Medical AI Disclaimer**: This analysis is for educational purposes only. Always consult healthcare professionals for medical decisions.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("Please enter a question about the image.")
        else:
            st.sidebar.error("❌ Models failed to load")
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error("**Model Loading Error**: Some models failed to load. This might be due to:")
            st.write("- Insufficient memory resources")
            st.write("- Network connectivity issues") 
            st.write("- Model compatibility problems")
            st.write("\n**Please try refreshing the page or contact support.**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif app_mode == "About":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("ℹ️ About Medical Vision AI Assistant")
        st.write("""
        This application combines advanced AI technologies to assist with medical image analysis:
        
        **🔍 Features:**
        - **Medical Image Analysis**: Upload medical images (X-rays, CT scans, MRIs) and ask questions
        - **Bilingual Support**: Ask questions in English or Arabic, get answers in both languages
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
        - **Languages**: English and Arabic
        
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
            
            # Display model information
            st.subheader("🧠 AI Models Used")
            st.write("- Medical VQA: sharawy53/final_diploma_blip-med-rad-arabic")
            st.write("- Translation: Helsinki-NLP/opus-mt models")
            
        except:
            st.write("- System information unavailable")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Medical AI Disclaimer:** This is a demonstration application for educational purposes. Always consult with qualified healthcare professionals for medical decisions, diagnosis, and treatment.")

if __name__ == "__main__":
    main()
