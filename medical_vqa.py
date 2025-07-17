import streamlit as st
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import re
import time
from deep_translator import GoogleTranslator

# Configure page
st.set_page_config(
    page_title="MediVision AI - Smart Medical Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Minimalist CSS Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables */
    :root {
        --primary-blue: #0ea5e9;
        --primary-teal: #14b8a6;
        --light-gray: #f8fafc;
        --medium-gray: #64748b;
        --dark-gray: #1e293b;
        --white: #ffffff;
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background-color: #f9fbfd;
        min-height: 100vh;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-teal) 100%);
        color: white;
        padding: 1.5rem 1rem;
        text-align: center;
        margin-bottom: 2rem;
        border-radius: 0 0 1.5rem 1.5rem;
        box-shadow: var(--shadow-sm);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Navigation Tabs */
    .nav-tabs {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .nav-tab {
        padding: 0.7rem 1.5rem;
        border-radius: 2rem;
        background: white;
        color: var(--primary-blue);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        border: 2px solid var(--primary-blue);
    }
    
    .nav-tab.active {
        background: var(--primary-blue);
        color: white;
    }
    
    /* Main Content Container */
    .content-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Image Section */
    .image-section {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        margin-bottom: 1.5rem;
    }
    
    /* Analysis Section */
    .analysis-section {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
    }
    
    /* Quick Questions */
    .quick-questions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .question-btn {
        background: #f0f9ff;
        border: 1px solid #e0f2fe;
        padding: 0.6rem 1rem;
        border-radius: 0.75rem;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.9rem;
    }
    
    .question-btn:hover {
        background: #e0f2fe;
        border-color: var(--primary-blue);
    }
    
    /* Result Boxes */
    .result-box {
        background: #f0fdf4;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 3px solid var(--primary-teal);
        margin: 1.5rem 0;
    }
    
    /* Translation Boxes */
    .translation-item {
        background: #f8fafc;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        border-left: 2px solid var(--primary-teal);
    }
    
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Language Badges */
    .language-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .english-badge {
        background: var(--primary-blue);
        color: white;
    }
    
    .arabic-badge {
        background: var(--primary-teal);
        color: white;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .nav-tab {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
        
        .content-container {
            padding: 0 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_medical_vqa_model():
    """Load medical VQA model with enhanced UI feedback"""
    try:
        model_name = "sharawy53/final_diploma_blip-med-rad-arabic"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"🚨 Error loading VQA model: {str(e)}")
        return None, None

def is_arabic(text):
    """Check if text contains Arabic characters"""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def translate_text(text, source_lang, target_lang, max_retries=3):
    """Translate text using deep-translator with enhanced error handling"""
    if not text.strip():
        return text, False
        
    try:
        for attempt in range(max_retries):
            try:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                translated_text = translator.translate(text)
                return translated_text, True
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        return text, False
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text, False

def analyze_medical_image(image, question, processor, model):
    """Analyze medical image with VQA"""
    try:
        inputs = processor(image, question, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_length=100, num_beams=5)
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"🚨 Error analyzing image: {str(e)}"

def ensure_arabic_answer(answer, source_lang='en', target_lang='ar'):
    """Ensure the answer is in Arabic, translate if necessary"""
    if is_arabic(answer):
        return answer, False
    
    try:
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
        st.session_state.lang = 'en'
    
    # Modern Header
    st.markdown('''
    <div class="main-header">
        <h1>🩺 MediVision AI</h1>
        <p>Advanced Medical Image Analysis & Multilingual Support</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Navigation Tabs
    tabs = ["🔬 Medical Analysis", "ℹ️ About"]
    active_tab = st.radio(
        "Navigation:", 
        tabs, 
        horizontal=True, 
        label_visibility="collapsed",
        index=0
    )
    
    # Main content container
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    if active_tab == "🔬 Medical Analysis":
        # Load models
        with st.spinner("🔄 Loading AI models..."):
            vqa_processor, vqa_model = load_medical_vqa_model()
        
        if vqa_processor and vqa_model:
            # Image Upload Section
            st.markdown('''
            <div class="image-section">
                <h3>📤 Upload Medical Image</h3>
            </div>
            ''', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a medical image...", 
                type=["jpg", "jpeg", "png", "bmp"],
                label_visibility="collapsed"
            )
            
            image_col, info_col = st.columns([2, 1])
            
            with image_col:
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Medical Image for Analysis", use_container_width=True)
            
            with info_col:
                if uploaded_file is not None:
                    st.markdown(f'''
                    <div style="margin-top: 1rem;">
                        <p><strong>📏 Dimensions:</strong> {image.size[0]} x {image.size[1]} pixels</p>
                        <p><strong>📁 Size:</strong> {round(uploaded_file.size / 1024, 1)} KB</p>
                        <p><strong>🎨 Format:</strong> {uploaded_file.type}</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            # Analysis Section
            st.markdown('''
            <div class="analysis-section">
                <h3>❓ Ask Medical Questions</h3>
            </div>
            ''', unsafe_allow_html=True)
            
            # Language Selection
            lang = st.radio(
                "Language:", 
                ["🇺🇸 English", "🇸🇦 العربية"],
                horizontal=True,
                index=0 if st.session_state.lang == 'en' else 1,
                key="lang_selector"
            )
            
            st.session_state.lang = 'en' if lang == "🇺🇸 English" else 'ar'
            
            # Suggested Questions
            questions = {
                "en": [
                    "What abnormalities do you see?",
                    "Are there any fractures visible?",
                    "Is this result normal or abnormal?",
                    "Describe the key medical findings",
                    "Any signs of infection present?",
                    "Is there a tumor or mass visible?",
                    "What is your diagnostic assessment?"
                ],
                "ar": [
                    "ما هي التشوهات التي تراها؟",
                    "هل هناك أي كسور مرئية؟",
                    "هل هذه النتيجة طبيعية أم غير طبيعية؟",
                    "صف النتائج الطبية الرئيسية",
                    "هل هناك أي علامات للعدوى؟",
                    "هل هناك ورم أو كتلة مرئية؟",
                    "ما هو تقييمك التشخيصي؟"
                ]
            }
            
            st.markdown('<div class="quick-questions">', unsafe_allow_html=True)
            for i, q in enumerate(questions[st.session_state.lang]):
                if st.button(q, key=f"q_{i}_{st.session_state.lang}"):
                    st.session_state.question = q
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Custom Question
            placeholder = "Type your medical question here..." if st.session_state.lang == 'en' else "اكتب سؤالك الطبي هنا..."
            question = st.text_area(
                "Your Question:", 
                value=st.session_state.get('question', ''),
                placeholder=placeholder,
                height=100,
                label_visibility="collapsed"
            )
            
            # Analyze Button
            if st.button("🔬 Analyze Medical Image", type="primary", use_container_width=True):
                if uploaded_file is None:
                    st.warning("Please upload a medical image first")
                elif not question:
                    st.warning("Please enter a question about the medical image")
                else:
                    # Translate question if needed
                    question_is_arabic = is_arabic(question)
                    
                    if st.session_state.lang == 'en' and question_is_arabic:
                        display_question_en, _ = translate_text(question, "ar", "en")
                        model_question = question
                        display_question_ar = question
                    elif st.session_state.lang == 'en' and not question_is_arabic:
                        model_question, _ = translate_text(question, "en", "ar")
                        display_question_en = question
                        display_question_ar = model_question
                    elif st.session_state.lang == 'ar' and not question_is_arabic:
                        model_question, _ = translate_text(question, "en", "ar")
                        display_question_en = question
                        display_question_ar = model_question
                    else:
                        display_question_en, _ = translate_text(question, "ar", "en")
                        model_question = question
                        display_question_ar = question
                    
                    # Add medical context
                    contextualized_question = get_medical_context(model_question)
                    
                    # Analyze image
                    with st.spinner("🧠 Analyzing your medical image..."):
                        arabic_answer = analyze_medical_image(image, contextualized_question, vqa_processor, vqa_model)
                    
                    # Ensure answer is in Arabic
                    arabic_answer_display, arabic_translated = ensure_arabic_answer(arabic_answer)
                    
                    # Translate to English
                    with st.spinner("🌐 Translating results..."):
                        english_answer, _ = translate_text(arabic_answer, "ar", "en")
                    
                    # Display results
                    st.markdown('''
                    <div class="result-box">
                        <h3>🔍 Medical Analysis Results</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Question display
                    st.markdown(f'''
                    <div class="translation-item">
                        <strong>Question:</strong> 
                        {display_question_en}
                        <span class="language-badge english-badge">EN</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.markdown(f'''
                    <div class="translation-item rtl-text">
                        <strong>السؤال:</strong> 
                        {display_question_ar}
                        <span class="language-badge arabic-badge">AR</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Answer display
                    st.markdown(f'''
                    <div class="translation-item">
                        <strong>Analysis:</strong> 
                        {english_answer}
                        <span class="language-badge english-badge">EN</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.markdown(f'''
                    <div class="translation-item rtl-text">
                        <strong>التحليل:</strong> 
                        {arabic_answer_display}
                        <span class="language-badge arabic-badge">AR</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Medical disclaimer
                    st.info("""
                    **⚠️ Medical Disclaimer**  
                    This AI analysis is for educational purposes only. Always consult with qualified healthcare 
                    professionals for medical decisions. AI responses may contain errors and should not replace 
                    professional medical judgment.
                    """)
        
        else:
            st.error("Failed to load AI models. Please try again later.")
    
    elif active_tab == "ℹ️ About":
        # About section with simplified layout
        st.markdown('''
        <div class="analysis-section">
            <h3>ℹ️ About MediVision AI</h3>
            <p>Advanced medical image analysis platform combining cutting-edge AI technologies with 
            multilingual support for healthcare professionals and medical students worldwide.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Features and Technology in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Core Features")
            st.markdown("""
            - 🩻 X-ray, CT, MRI & Ultrasound analysis
            - 🌍 English/Arabic bilingual support
            - 🧠 Specialized medical AI models
            - 🎯 Context-aware understanding
            - 💬 Natural language interaction
            - 📊 Detailed medical insights
            """)
            
        with col2:
            st.subheader("🛠️ Technology")
            st.markdown("""
            - 🤖 BLIP Vision-Language Model
            - 🔥 PyTorch Deep Learning
            - 🌐 Google Translator API
            - 🚀 Streamlit Framework
            - 🐍 Python Backend
            - 💾 Hugging Face Transformers
            """)
        
        # Medical disclaimer
        st.info("""
        **🩺 Professional Medical Disclaimer**  
        This is a demonstration application for educational and research purposes only.  
        Always consult with qualified healthcare professionals for medical decisions, diagnosis, and treatment. 
        AI-generated analysis should never replace professional medical judgment.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close content-container

if __name__ == "__main__":
    main()
