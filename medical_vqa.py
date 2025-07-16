import streamlit as st
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import re
import requests
import json

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

def translate_text(text, source_lang, target_lang):
    """Translate text using external API (LibreTranslate)"""
    if not text.strip():
        return text
        
    try:
        # Use public LibreTranslate API
        url = "https://libretranslate.com/translate"
        payload = {
            'q': text,
            'source': source_lang,
            'target': target_lang,
            'format': 'text'
        }
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        
        result = response.json()
        return result.get('translatedText', text)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

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
                    st.image(image, caption="Uploaded Medical Image", use_column_width=True)
                    st.info(f"Image size: {image.size[0]}x{image.size[1]} pixels")
                
                with col2:
                    # Language selector
                    st.markdown('<div class="radio-horizontal">', unsafe_allow_html=True)
                    lang = st.radio("Select Language:", 
                                   ["English", "العربية"],
                                   horizontal=True,
                                   index=0 if st.session_state.lang == 'english' else 1,
                                   label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.session_state.lang = 'english' if lang == "English" else 'arabic'
                    
                    # Question input
                    st.subheader("Ask a Medical Question")
                    
                    # Suggested questions
                    st.write("**Suggested Questions:**")
                    
                    # Questions in both languages
                    questions = {
                        "english": [
                            "What abnormalities do you see?",
                            "Are there any fractures?",
                            "Is this result normal or abnormal?",
                            "Describe the key findings",
                            "Any signs of infection?",
                            "Is there a tumor visible?",
                            "What is the diagnosis?"
                        ],
                        "arabic": [
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
                    placeholder = "Type your question here..." if st.session_state.lang == 'english' else "اكتب سؤالك هنا..."
                    question = st.text_area("Your question:", 
                                           value=st.session_state.get('question', ''),
                                           placeholder=placeholder,
                                           height=100)
                    
                    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                        if question:
                            # Translate question to Arabic if needed (model requires Arabic)
                            original_question = question
                            
                            # If question is in English, translate to Arabic for the model
                            if not is_arabic(question):
                                with st.spinner("Translating question to Arabic..."):
                                    question = translate_text(question, "en", "ar")
                            
                            # Add medical context
                            contextualized_question = get_medical_context(question)
                            
                            # Analyze image
                            with st.spinner("Analyzing medical image..."):
                                arabic_answer = analyze_medical_image(image, contextualized_question, 
                                                                     vqa_processor, vqa_model)
                            
                            # Translate answer to English
                            with st.spinner("Translating answer to English..."):
                                english_answer = translate_text(arabic_answer, "ar", "en")
                            
                            # Display results
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.subheader("🔍 Analysis Result")
                            
                            # Display question in both languages
                            st.markdown(f"""
                            <div class="translation-box">
                                <div>
                                    <strong>Your Question:</strong> 
                                    <span>{original_question}</span>
                                    <span class="language-badge english-badge">EN</span>
                                </div>
                                <div style="margin-top: 0.5rem;">
                                    <strong>سؤالك:</strong> 
                                    <span>{question}</span>
                                    <span class="language-badge arabic-badge">AR</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display answer in both languages
                            st.markdown(f"""
                            <div class="translation-box">
                                <div>
                                    <strong>Answer:</strong> 
                                    <span>{english_answer}</span>
                                    <span class="language-badge english-badge">EN</span>
                                </div>
                                <div style="margin-top: 0.5rem;">
                                    <strong>الإجابة:</strong> 
                                    <span>{arabic_answer}</span>
                                    <span class="language-badge arabic-badge">AR</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add confidence disclaimer
                            st.caption("⚠️ **Medical AI Disclaimer**: This analysis is for educational purposes only. Always consult healthcare professionals for medical decisions.")
                            st.markdown('</div>', unsafe_allow_html=True)
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
        
        **📋 Supported:**
        - **Image Types**: X-rays, CT scans, MRIs, ultrasounds
        - **Formats**: JPG, PNG, BMP
        - **Languages**: English and Arabic interface
        
        **⚠️ Important Disclaimers:**
        - This tool is for **educational and research purposes only**
        - **NOT a substitute** for professional medical diagnosis
        - Always consult qualified healthcare professionals
        - AI responses may contain errors or limitations
        
        **Note on Translation**: 
        The translation functionality uses LibreTranslate API for Arabic-English translation.
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
