import streamlit as st
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import re
import time
from deep_translator import GoogleTranslator

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
    .translation-item {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        background-color: #f8fafc;
    }
    .quick-questions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .language-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .english-badge {
        background-color: #3b82f6;
        color: white;
    }
    .arabic-badge {
        background-color: #10b981;
        color: white;
    }
    .radio-horizontal {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .model-status {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .status-success {
        background-color: #dcfce7;
        color: #166534;
    }
    .status-error {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    .error-details {
        background-color: #fffbeb;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f59e0b;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    .warning-badge {
        background-color: #f59e0b;
        color: white;
        padding: 0.1rem 0.4rem;
        border-radius: 0.25rem;
        font-size: 0.7rem;
        margin-left: 0.5rem;
    }
    .info-box {
        background-color: #dbeafe;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
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

def is_arabic(text):
    """Check if text contains Arabic characters"""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def translate_text(text, source_lang, target_lang, max_retries=3):
    """Translate text using deep-translator"""
    if not text.strip():
        return text, False
        
    try:
        # Add retry mechanism
        for attempt in range(max_retries):
            try:
                # Use GoogleTranslator for reliable translation
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                translated_text = translator.translate(text)
                return translated_text, True
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        return text, False
    except Exception as e:
        # Detailed error reporting
        error_details = f"**Translation Error Details:**\n"
        error_details += f"- **Error Type**: {type(e).__name__}\n"
        error_details += f"- **Message**: {str(e)}\n"
        
        st.markdown(f'<div class="error-details">', unsafe_allow_html=True)
        st.error(f"**Translation Error**: {str(e)}")
        st.markdown(error_details, unsafe_allow_html=True)
        
        # Troubleshooting tips
        st.warning("**Troubleshooting Tips:**")
        st.write("1. Verify your text doesn't contain special characters")
        st.write("2. Ensure text length is under 5000 characters")
        st.write("3. Check language codes (en for English, ar for Arabic)")
        st.write("4. Try again after a few seconds")
        st.markdown('</div>', unsafe_allow_html=True)
        
        return text, False

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

def ensure_arabic_answer(answer, source_lang='en', target_lang='ar'):
    """Ensure the answer is in Arabic, translate if necessary"""
    if is_arabic(answer):
        return answer, False  # Already in Arabic, no translation needed
    
    try:
        # Translate to Arabic if not already in Arabic
        translated, success = translate_text(answer, source_lang, target_lang)
        if success:
            return translated, True
        return answer, False
    except:
        return answer, False

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
        st.session_state.lang = 'en'  # Use ISO codes
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Vision AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", 
                                   ["Medical Image Analysis", "About"])
    
    # Add model status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Models Status")
    
    # Translation info
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 Translation Info")
    st.sidebar.markdown("""
    <div class="info-box">
        Using Google Translator service.<br>
        No API key required.<br>
        Translations are reliable and accurate.
    </div>
    """, unsafe_allow_html=True)
    
    if app_mode == "Medical Image Analysis":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📊 Medical Image Analysis")
        st.write("Upload a medical image and ask questions about it using AI-powered visual question answering.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load models
        with st.spinner("Loading medical AI model..."):
            vqa_processor, vqa_model = load_medical_vqa_model()
        
        # Display model status
        model_status = []
        if vqa_processor and vqa_model:
            model_status.append(('Medical VQA Model', '✅ Ready', 'status-success'))
            st.sidebar.markdown(f'<div class="model-status status-success">Medical VQA Model: ✅ Ready</div>', unsafe_allow_html=True)
        else:
            model_status.append(('Medical VQA Model', '❌ Failed', 'status-error'))
            st.sidebar.markdown(f'<div class="model-status status-error">Medical VQA Model: ❌ Failed</div>', unsafe_allow_html=True)
        
        st.sidebar.markdown(f'<div class="model-status status-success">Translation Service: ✅ Ready</div>', unsafe_allow_html=True)
        
        if vqa_processor and vqa_model:
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
                    # Language selector
                    st.markdown('<div class="radio-horizontal">', unsafe_allow_html=True)
                    lang = st.radio("Select Language:", 
                                   ["English", "العربية"],
                                   horizontal=True,
                                   index=0 if st.session_state.lang == 'en' else 1,
                                   label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.session_state.lang = 'en' if lang == "English" else 'ar'
                    
                    # Question input
                    st.subheader("Ask a Medical Question")
                    
                    # Suggested questions
                    st.write("**Suggested Questions:**")
                    
                    # Questions in both languages
                    questions = {
                        "en": [
                            "What abnormalities do you see?",
                            "Are there any fractures?",
                            "Is this result normal or abnormal?",
                            "Describe the key findings",
                            "Any signs of infection?",
                            "Is there a tumor visible?",
                            "What is the diagnosis?"
                        ],
                        "ar": [
                            "ما هي التشوهات التي تراها؟",
                            "هل هناك أي كسور؟",
                            "هل هذه النتيجة طبيعية أم غير طبيعية؟",
                            "صف النتائج الرئيسية",
                            "هل هناك أي علامات للعدوى؟",
                            "هل هناك ورم مرئي؟",
                            "ما هو التشخيص؟"
                        ]
                    }
                    
                    # Display suggested questions based on selected language
                    cols = st.columns(2)
                    for i, q in enumerate(questions[st.session_state.lang]):
                        col = cols[i % 2]
                        if col.button(q, key=f"q_{i}_{st.session_state.lang}", use_container_width=True):
                            st.session_state.question = q
                    
                    # Custom question input
                    placeholder = "Type your question here..." if st.session_state.lang == 'en' else "اكتب سؤالك هنا..."
                    question = st.text_area("Your question:", 
                                           value=st.session_state.get('question', ''),
                                           placeholder=placeholder,
                                           height=100)
                    
                    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                        if question:
                            # Store original question
                            original_question = question
                            
                            # Determine if question is Arabic or English
                            question_is_arabic = is_arabic(question)
                            
                            # Prepare variables for display
                            display_question_en = ""
                            display_question_ar = ""
                            
                            # Case 1: User interface is English but question is in Arabic
                            if st.session_state.lang == 'en' and question_is_arabic:
                                # Translate Arabic question to English for display
                                with st.spinner("Translating question to English..."):
                                    display_question_en, success = translate_text(
                                        question, "ar", "en"
                                    )
                                    if not success:
                                        display_question_en = question + " [Auto]"
                                
                                # For model, we can use the original Arabic question
                                model_question = question
                                display_question_ar = question
                            
                            # Case 2: User interface is English and question is in English
                            elif st.session_state.lang == 'en' and not question_is_arabic:
                                # Translate English question to Arabic for the model
                                with st.spinner("Translating question to Arabic..."):
                                    model_question, success = translate_text(
                                        question, "en", "ar"
                                    )
                                    if not success:
                                        model_question = question + " [Auto]"
                                
                                display_question_en = question
                                display_question_ar = model_question
                            
                            # Case 3: User interface is Arabic but question is in English
                            elif st.session_state.lang == 'ar' and not question_is_arabic:
                                # Translate English question to Arabic for display and model
                                with st.spinner("Translating question to Arabic..."):
                                    model_question, success = translate_text(
                                        question, "en", "ar"
                                    )
                                    if not success:
                                        model_question = question + " [Auto]"
                                
                                # For English display, we can use the original question
                                display_question_en = question
                                display_question_ar = model_question
                            
                            # Case 4: User interface is Arabic and question is in Arabic
                            else:  # st.session_state.lang == 'ar' and question_is_arabic
                                # Translate Arabic question to English for display
                                with st.spinner("Translating question to English..."):
                                    display_question_en, success = translate_text(
                                        question, "ar", "en"
                                    )
                                    if not success:
                                        display_question_en = question + " [Auto]"
                                
                                model_question = question
                                display_question_ar = question
                            
                            # Add medical context to the question that will be sent to the model
                            contextualized_question = get_medical_context(model_question)
                            
                            # Analyze image
                            with st.spinner("Analyzing medical image..."):
                                arabic_answer = analyze_medical_image(image, contextualized_question, 
                                                                     vqa_processor, vqa_model)
                            
                            # Ensure the answer is properly in Arabic for display
                            arabic_answer_display, arabic_translated = ensure_arabic_answer(arabic_answer)
                            
                            # Always translate the answer to English
                            with st.spinner("Translating answer to English..."):
                                english_answer, success = translate_text(
                                    arabic_answer, "ar", "en"
                                )
                                if not success:
                                    english_answer = arabic_answer + " [Auto]"
                            
                            # Display results
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.subheader("🔍 Analysis Result")
                            
                            # Display question in both languages - FIXED FOR ARABIC
                            st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                            
                            # English question display
                            st.markdown('<div class="translation-item">', unsafe_allow_html=True)
                            st.markdown(f'<strong>Your Question:</strong> <span>{display_question_en}</span> <span class="language-badge english-badge">EN</span>', unsafe_allow_html=True)
                            if "[Auto]" in display_question_en:
                                st.markdown('<span class="warning-badge">Auto</span>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Arabic question display with RTL support
                            st.markdown('<div class="translation-item rtl-text">', unsafe_allow_html=True)
                            st.markdown(f'<strong>سؤالك:</strong> <span>{display_question_ar}</span> <span class="language-badge arabic-badge">AR</span>', unsafe_allow_html=True)
                            if "[Auto]" in display_question_ar:
                                st.markdown('<span class="warning-badge">Auto</span>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)  # Close translation-box
                            
                            # Display answer in both languages
                            st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                            
                            # English answer display
                            st.markdown('<div class="translation-item">', unsafe_allow_html=True)
                            st.markdown(f'<strong>Answer:</strong> <span>{english_answer}</span> <span class="language-badge english-badge">EN</span>', unsafe_allow_html=True)
                            if "[Auto]" in english_answer:
                                st.markdown('<span class="warning-badge">Auto</span>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Arabic answer display with RTL support
                            st.markdown('<div class="translation-item rtl-text">', unsafe_allow_html=True)
                            st.markdown(f'<strong>الإجابة:</strong> <span>{arabic_answer_display}</span> <span class="language-badge arabic-badge">AR</span>', unsafe_allow_html=True)
                            if arabic_translated:
                                st.markdown('<span class="warning-badge">مترجم</span>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)  # Close translation-box
                            
                            # Add confidence disclaimer
                            st.caption("⚠️ **Medical AI Disclaimer**: This analysis is for educational purposes only. Always consult healthcare professionals for medical decisions.")
                            st.markdown('</div>', unsafe_allow_html=True)  # Close result-box
                        else:
                            st.warning("Please enter a question about the image.")
        else:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error("**Model Loading Error**: The medical VQA model failed to load. This might be due to:")
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
        - **AI-Powered**: Uses state-of-the-art vision models
        - **Medical Context**: Specialized for medical terminology and scenarios
        
        **🛠️ Technologies Used:**
        - **Streamlit**: Web interface framework
        - **BLIP**: Vision-language model for image question answering
        - **PyTorch**: Deep learning framework
        - **Transformers**: Hugging Face model library
        - **Deep Translator**: Reliable translation service
        
        **📋 Supported:**
        - **Image Types**: X-rays, CT scans, MRIs, ultrasounds
        - **Formats**: JPG, PNG, BMP
        - **Languages**: English and Arabic interface
        
        **⚠️ Important Disclaimers:**
        - This tool is for **educational and research purposes only**
        - **NOT a substitute** for professional medical diagnosis
        - Always consult qualified healthcare professionals
        - AI responses may contain errors or limitations
        - Medical terms may be translated for better understanding
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
            
        except:
            st.write("- System information unavailable")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Medical AI Disclaimer:** This is a demonstration application for educational purposes. Always consult with qualified healthcare professionals for medical decisions, diagnosis, and treatment.")

if __name__ == "__main__":
    main()
