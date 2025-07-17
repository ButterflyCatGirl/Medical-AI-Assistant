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
    initial_sidebar_state="expanded"
)

# Modern E-Health CSS Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables */
    :root {
        --primary-blue: #0ea5e9;
        --primary-teal: #14b8a6;
        --primary-mint: #10d9c4;
        --secondary-purple: #8b5cf6;
        --accent-orange: #f97316;
        --success-green: #22c55e;
        --warning-yellow: #eab308;
        --error-red: #ef4444;
        --dark-blue: #1e40af;
        --light-gray: #f8fafc;
        --medium-gray: #64748b;
        --dark-gray: #1e293b;
        --white: #ffffff;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1);
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        min-height: 100vh;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-teal) 100%);
        color: white;
        padding: 2rem;
        text-align: center;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 2rem 2rem;
        box-shadow: var(--shadow-xl);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, var(--white) 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #e2e8f0;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-blue), var(--primary-teal), var(--primary-mint));
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-blue);
    }
    
    .feature-card h3 {
        color: var(--dark-blue);
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status Cards */
    .status-card {
        background: var(--white);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        border-left: 4px solid var(--success-green);
        transition: all 0.3s ease;
    }
    
    .status-card:hover {
        box-shadow: var(--shadow-md);
    }
    
    .status-success {
        border-left-color: var(--success-green);
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    
    .status-error {
        border-left-color: var(--error-red);
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }
    
    .status-warning {
        border-left-color: var(--warning-yellow);
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    /* Result Boxes */
    .result-box {
        background: linear-gradient(135deg, var(--white) 0%, #f0fdf4 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid var(--success-green);
        margin: 2rem 0;
        box-shadow: var(--shadow-lg);
        position: relative;
    }
    
    .result-box::before {
        content: '✨';
        position: absolute;
        top: -10px;
        right: 20px;
        background: var(--success-green);
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }
    
    /* Translation Boxes */
    .translation-box {
        background: linear-gradient(135deg, var(--white) 0%, #f0f9ff 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid var(--primary-blue);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    .translation-item {
        background: rgba(255, 255, 255, 0.8);
        padding: 1rem;
        margin: 0.75rem 0;
        border-radius: 0.5rem;
        border-left: 3px solid var(--primary-teal);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .translation-item:hover {
        background: rgba(255, 255, 255, 0.95);
        transform: translateX(5px);
    }
    
    /* Language Badges */
    .language-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 2rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    .english-badge {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--dark-blue) 100%);
        color: white;
        box-shadow: var(--shadow-sm);
    }
    
    .arabic-badge {
        background: linear-gradient(135deg, var(--primary-teal) 0%, var(--success-green) 100%);
        color: white;
        box-shadow: var(--shadow-sm);
    }
    
    .warning-badge {
        background: linear-gradient(135deg, var(--warning-yellow) 0%, var(--accent-orange) 100%);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        font-weight: 500;
        margin-left: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Quick Questions */
    .quick-questions {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    
    .question-btn {
        background: linear-gradient(135deg, var(--white) 0%, #f1f5f9 100%);
        border: 2px solid #e2e8f0;
        padding: 1rem;
        border-radius: 0.75rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: left;
        font-weight: 500;
        color: var(--dark-gray);
    }
    
    .question-btn:hover {
        border-color: var(--primary-blue);
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-teal) 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        border-left: 4px solid var(--primary-blue);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    .error-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid var(--error-red);
        margin: 1rem 0;
        box-shadow: var(--shadow-md);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fed7aa 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        border-left: 4px solid var(--warning-yellow);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    /* RTL Text Support */
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar Enhancements */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, var(--white) 0%, #f8fafc 100%);
    }
    
    /* Model Status Indicators */
    .model-status {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-weight: 500;
        gap: 0.5rem;
    }
    
    .model-status.status-success {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #166534;
        border: 1px solid #22c55e;
    }
    
    .model-status.status-error {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #b91c1c;
        border: 1px solid #ef4444;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid var(--primary-blue);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 0.5rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .feature-card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .quick-questions {
            grid-template-columns: 1fr;
        }
    }
    
    /* Custom Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-teal) 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, var(--dark-blue) 0%, var(--primary-blue) 100%);
    }
    
    /* File Uploader Enhancement */
    .uploadedFile {
        border: 2px dashed var(--primary-blue);
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, var(--white) 0%, #f0f9ff 100%);
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: var(--primary-teal);
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    }
    
    /* Medical Icons */
    .medical-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        color: var(--primary-teal);
    }
    
    /* Disclaimer */
    .medical-disclaimer {
        background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 50%, #fb923c 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 4px solid var(--accent-orange);
        margin: 2rem 0;
        box-shadow: var(--shadow-md);
        position: relative;
    }
    
    .medical-disclaimer::before {
        content: '⚠️';
        position: absolute;
        top: -10px;
        left: 20px;
        background: var(--accent-orange);
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
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
        st.markdown(f'''
        <div class="error-box">
            <h4>🚨 Translation Error</h4>
            <p><strong>Error Type:</strong> {type(e).__name__}</p>
            <p><strong>Message:</strong> {str(e)}</p>
            <div class="warning-box">
                <h5>💡 Troubleshooting Tips:</h5>
                <ul>
                    <li>Verify text doesn't contain special characters</li>
                    <li>Ensure text length is under 5000 characters</li>
                    <li>Check language codes (en for English, ar for Arabic)</li>
                    <li>Try again after a few seconds</li>
                </ul>
            </div>
        </div>
        ''', unsafe_allow_html=True)
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
    
    # Enhanced Sidebar
    st.sidebar.markdown('''
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #0ea5e9 0%, #14b8a6 100%); border-radius: 1rem; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">🏥 Navigation</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    app_mode = st.sidebar.selectbox(
        "Choose Application Mode", 
        ["🔬 Medical Image Analysis", "ℹ️ About & Information"],
        help="Select the functionality you want to use"
    )
    
    # Enhanced Model Status in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown('''
    <div style="text-align: center; padding: 0.5rem;">
        <h3 style="color: #1e40af;">🤖 AI System Status</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    # Translation Service Status
    st.sidebar.markdown('''
    <div class="info-box">
        <h4>🌐 Translation Service</h4>
        <p><strong>Provider:</strong> Google Translator</p>
        <p><strong>Status:</strong> <span style="color: #22c55e;">✅ Active</span></p>
        <p><strong>Languages:</strong> Arabic ↔ English</p>
        <p><strong>Reliability:</strong> High Accuracy</p>
    </div>
    ''', unsafe_allow_html=True)
    
    if app_mode == "🔬 Medical Image Analysis":
        st.markdown('''
        <div class="feature-card">
            <h3>🔬 Medical Image Analysis</h3>
            <p>Upload medical images (X-rays, CT scans, MRIs, Ultrasounds) and get AI-powered analysis with multilingual support. Our advanced vision models provide detailed insights about your medical imagery.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Load models with enhanced feedback
        with st.spinner("🔄 Loading AI models..."):
            vqa_processor, vqa_model = load_medical_vqa_model()
        
        # Display enhanced model status
        if vqa_processor and vqa_model:
            st.sidebar.markdown('''
            <div class="model-status status-success">
                <span class="loading-spinner" style="border-top-color: #22c55e;"></span>
                Medical VQA Model: ✅ Ready
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('''
            <div class="model-status status-error">
                ❌ Medical VQA Model: Failed to Load
            </div>
            ''', unsafe_allow_html=True)
        
        if vqa_processor and vqa_model:
            # Enhanced File Upload Section
            st.markdown('''
            <div class="feature-card">
                <h3>📤 Upload Medical Image</h3>
                <p>Supported formats: JPG, PNG, BMP | Maximum size: 200MB</p>
            </div>
            ''', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a medical image...", 
                type=["jpg", "jpeg", "png", "bmp"],
                help="Upload X-rays, CT scans, MRIs, or ultrasound images for AI analysis"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                
                # Enhanced Image Display
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown('''
                    <div class="feature-card">
                        <h3>🖼️ Uploaded Image</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.image(image, caption="Medical Image for Analysis", use_container_width=True)
                    
                    st.markdown(f'''
                    <div class="info-box">
                        <p><strong>📏 Image Dimensions:</strong> {image.size[0]} x {image.size[1]} pixels</p>
                        <p><strong>📁 File Size:</strong> {round(uploaded_file.size / 1024, 1)} KB</p>
                        <p><strong>🎨 Format:</strong> {uploaded_file.type}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    # Enhanced Language Selector
                    st.markdown('''
                    <div class="feature-card">
                        <h3>🌍 Language Settings</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    lang = st.radio(
                        "Select Interface Language:", 
                        ["🇺🇸 English", "🇸🇦 العربية"],
                        horizontal=True,
                        index=0 if st.session_state.lang == 'en' else 1,
                        help="Choose your preferred language for questions and results"
                    )
                    
                    st.session_state.lang = 'en' if lang == "🇺🇸 English" else 'ar'
                    
                    # Enhanced Question Input Section
                    st.markdown('''
                    <div class="feature-card">
                        <h3>❓ Ask Medical Questions</h3>
                        <p>Click on suggested questions or type your own custom question</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Enhanced Suggested Questions
                    questions = {
                        "en": [
                            "🔍 What abnormalities do you see?",
                            "🦴 Are there any fractures visible?",
                            "✅ Is this result normal or abnormal?",
                            "📝 Describe the key medical findings",
                            "🦠 Any signs of infection present?",
                            "🎯 Is there a tumor or mass visible?",
                            "🩺 What is your diagnostic assessment?"
                        ],
                        "ar": [
                            "🔍 ما هي التشوهات التي تراها؟",
                            "🦴 هل هناك أي كسور مرئية؟",
                            "✅ هل هذه النتيجة طبيعية أم غير طبيعية؟",
                            "📝 صف النتائج الطبية الرئيسية",
                            "🦠 هل هناك أي علامات للعدوى؟",
                            "🎯 هل هناك ورم أو كتلة مرئية؟",
                            "🩺 ما هو تقييمك التشخيصي؟"
                        ]
                    }
                    
                    st.markdown('<div class="quick-questions">', unsafe_allow_html=True)
                    for i, q in enumerate(questions[st.session_state.lang]):
                        if st.button(q, key=f"q_{i}_{st.session_state.lang}", use_container_width=True):
                            st.session_state.question = q.split(' ', 1)[1]  # Remove emoji
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Enhanced Custom Question Input
                    placeholder = "Type your medical question here..." if st.session_state.lang == 'en' else "اكتب سؤالك الطبي هنا..."
                    question = st.text_area(
                        "✍️ Custom Question:", 
                        value=st.session_state.get('question', ''),
                        placeholder=placeholder,
                        height=100,
                        help="Ask specific questions about the medical image"
                    )
                    
                    # Enhanced Analyze Button
                    if st.button("🔬 Analyze Medical Image", type="primary", use_container_width=True):
                        if question:
                            original_question = question
                            question_is_arabic = is_arabic(question)
                            
                            display_question_en = ""
                            display_question_ar = ""
                            
                            # Enhanced translation logic with better UI feedback
                            if st.session_state.lang == 'en' and question_is_arabic:
                                with st.spinner("🔄 Translating question to English..."):
                                    display_question_en, success = translate_text(question, "ar", "en")
                                    if not success:
                                        display_question_en = question + " [Auto]"
                                model_question = question
                                display_question_ar = question
                            
                            elif st.session_state.lang == 'en' and not question_is_arabic:
                                with st.spinner("🔄 Translating question to Arabic..."):
                                    model_question, success = translate_text(question, "en", "ar")
                                    if not success:
                                        model_question = question + " [Auto]"
                                display_question_en = question
                                display_question_ar = model_question
                            
                            elif st.session_state.lang == 'ar' and not question_is_arabic:
                                with st.spinner("🔄 Translating question to Arabic..."):
                                    model_question, success = translate_text(question, "en", "ar")
                                    if not success:
                                        model_question = question + " [Auto]"
                                display_question_en = question
                                display_question_ar = model_question
                            
                            else:
                                with st.spinner("🔄 Translating question to English..."):
                                    display_question_en, success = translate_text(question, "ar", "en")
                                    if not success:
                                        display_question_en = question + " [Auto]"
                                model_question = question
                                display_question_ar = question
                            
                            contextualized_question = get_medical_context(model_question)
                            
                            # Enhanced Analysis with better feedback
                            with st.spinner("🧠 AI is analyzing your medical image..."):
                                arabic_answer = analyze_medical_image(image, contextualized_question, vqa_processor, vqa_model)
                            
                            arabic_answer_display, arabic_translated = ensure_arabic_answer(arabic_answer)
                            
                            with st.spinner("🌐 Translating results to English..."):
                                english_answer, success = translate_text(arabic_answer, "ar", "en")
                                if not success:
                                    english_answer = arabic_answer + " [Auto]"
                            
                            # Enhanced Results Display
                            st.markdown('''
                            <div class="result-box">
                                <h2>🔍 Medical Analysis Results</h2>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            # Enhanced Question Display
                            st.markdown('''
                            <div class="translation-box">
                                <h4>❓ Your Medical Question</h4>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            st.markdown(f'''
                            <div class="translation-item">
                                <strong>🇺🇸 English Question:</strong> 
                                <span>{display_question_en}</span> 
                                <span class="language-badge english-badge">EN</span>
                                {"<span class='warning-badge'>Auto Translated</span>" if "[Auto]" in display_question_en else ""}
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            st.markdown(f'''
                            <div class="translation-item rtl-text">
                                <strong>🇸🇦 السؤال بالعربية:</strong> 
                                <span>{display_question_ar}</span> 
                                <span class="language-badge arabic-badge">AR</span>
                                {"<span class='warning-badge'>مترجم تلقائياً</span>" if "[Auto]" in display_question_ar else ""}
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            # Enhanced Answer Display
                            st.markdown('''
                            <div class="translation-box">
                                <h4>🩺 AI Medical Analysis</h4>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            st.markdown(f'''
                            <div class="translation-item">
                                <strong>🇺🇸 English Analysis:</strong> 
                                <span>{english_answer}</span> 
                                <span class="language-badge english-badge">EN</span>
                                {"<span class='warning-badge'>Auto Translated</span>" if "[Auto]" in english_answer else ""}
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            st.markdown(f'''
                            <div class="translation-item rtl-text">
                                <strong>🇸🇦 التحليل بالعربية:</strong> 
                                <span>{arabic_answer_display}</span> 
                                <span class="language-badge arabic-badge">AR</span>
                                {"<span class='warning-badge'>مترجم</span>" if arabic_translated else ""}
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            # Enhanced Medical Disclaimer
                            st.markdown('''
                            <div class="medical-disclaimer">
                                <h4>⚠️ Important Medical Disclaimer</h4>
                                <p><strong>This AI analysis is for educational and research purposes only.</strong></p>
                                <ul>
                                    <li>🩺 NOT a substitute for professional medical diagnosis</li>
                                    <li>👨‍⚕️ Always consult qualified healthcare professionals</li>
                                    <li>🔬 AI responses may contain errors or limitations</li>
                                    <li>📚 Use this tool as a learning aid, not for medical decisions</li>
                                </ul>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown('''
                            <div class="warning-box">
                                <h4>⚠️ Input Required</h4>
                                <p>Please enter a question about the medical image to proceed with analysis.</p>
                            </div>
                            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="error-box">
                <h3>🚨 AI Model Loading Error</h3>
                <p>The medical VQA model failed to load. This might be due to:</p>
                <ul>
                    <li>💾 Insufficient memory resources</li>
                    <li>🌐 Network connectivity issues</li> 
                    <li>⚙️ Model compatibility problems</li>
                    <li>🔧 System configuration issues</li>
                </ul>
                <div class="info-box">
                    <h4>💡 Troubleshooting Steps:</h4>
                    <ol>
                        <li>Refresh the page and try again</li>
                        <li>Check your internet connection</li>
                        <li>Clear your browser cache</li>
                        <li>Contact technical support if issues persist</li>
                    </ol>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    elif app_mode == "ℹ️ About & Information":
        st.markdown('''
        <div class="feature-card">
            <h3>ℹ️ About MediVision AI</h3>
            <p>Advanced medical image analysis platform combining cutting-edge AI technologies with multilingual support for healthcare professionals and medical students worldwide.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Enhanced Features Section
        st.markdown('''
        <div class="feature-card">
            <h3>🔍 Core Features</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div class="status-card status-success">
                    <h4>🩻 Medical Image Analysis</h4>
                    <p>Upload X-rays, CT scans, MRIs, and ultrasounds for AI-powered analysis</p>
                </div>
                <div class="status-card status-success">
                    <h4>🌍 Bilingual Support</h4>
                    <p>Ask questions in English or Arabic, receive answers in both languages</p>
                </div>
                <div class="status-card status-success">
                    <h4>🧠 AI-Powered Analysis</h4>
                    <p>State-of-the-art vision-language models specialized for medical imaging</p>
                </div>
                <div class="status-card status-success">
                    <h4>🎯 Medical Context</h4>
                    <p>Specialized understanding of medical terminology and clinical scenarios</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Enhanced Technology Stack
        st.markdown('''
        <div class="feature-card">
            <h3>🛠️ Technology Stack</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div class="info-box">
                    <h4>🖥️ Frontend Framework</h4>
                    <p><strong>Streamlit:</strong> Modern web interface</p>
                </div>
                <div class="info-box">
                    <h4>🤖 AI Models</h4>
                    <p><strong>BLIP:</strong> Vision-language model for VQA</p>
                </div>
                <div class="info-box">
                    <h4>🔥 Deep Learning</h4>
                    <p><strong>PyTorch:</strong> Neural network framework</p>
                </div>
                <div class="info-box">
                    <h4>🌐 Translation</h4>
                    <p><strong>Google Translator:</strong> Multi-language support</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Enhanced Supported Formats
        st.markdown('''
        <div class="feature-card">
            <h3>📋 Supported Medical Imaging</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div class="status-card status-success">
                    <h4>🩻 X-Ray Imaging</h4>
                    <p>Chest, bone, dental X-rays</p>
                </div>
                <div class="status-card status-success">
                    <h4>🧠 CT Scans</h4>
                    <p>Brain, chest, abdominal CT</p>
                </div>
                <div class="status-card status-success">
                    <h4>🔬 MRI Scans</h4>
                    <p>All anatomical regions</p>
                </div>
                <div class="status-card status-success">
                    <h4>📡 Ultrasound</h4>
                    <p>Obstetric, cardiac, abdominal</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Enhanced System Information
        st.markdown('''
        <div class="feature-card">
            <h3>🔧 System Information</h3>
        </div>
        ''', unsafe_allow_html=True)
        
        try:
            import torch
            st.markdown(f'''
            <div class="info-box">
                <p><strong>🔥 PyTorch Version:</strong> {torch.__version__}</p>
                <p><strong>💻 Computing Device:</strong> {'🚀 CUDA GPU Acceleration' if torch.cuda.is_available() else '🖥️ CPU Processing'}</p>
                <p><strong>🌐 Streamlit Version:</strong> {st.__version__}</p>
                <p><strong>🧠 Medical VQA Model:</strong> sharawy53/final_diploma_blip-med-rad-arabic</p>
                <p><strong>🌍 Translation Service:</strong> Google Translator API</p>
            </div>
            ''', unsafe_allow_html=True)
        except:
            st.markdown('''
            <div class="warning-box">
                <p>⚠️ System information temporarily unavailable</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown('''
    <div class="medical-disclaimer">
        <h4>🩺 Professional Medical Disclaimer</h4>
        <p><strong>This is a demonstration application for educational and research purposes only.</strong></p>
        <p>Always consult with qualified healthcare professionals for medical decisions, diagnosis, and treatment. AI-generated analysis should never replace professional medical judgment.</p>
        <p style="text-align: center; margin-top: 1rem;">
            <strong>© 2024 MediVision AI - Advancing Healthcare Through Technology</strong>
        </p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
