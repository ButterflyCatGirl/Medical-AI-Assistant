import streamlit as st
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import re
import time
from deep_translator import GoogleTranslator

# Configure page
st.set_page_config(
    page_title="Medical Vision AI - Egypt",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Medical UI Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables - Medical Theme */
    :root {
        --medical-blue: #0ea5e9;
        --medical-teal: #14b8a6;
        --medical-green: #10b981;
        --egypt-red: #ce1126;
        --egypt-black: #000000;
        --light-bg: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --border-radius: 12px;
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        min-height: 100vh;
    }
    
    /* Top Navigation Header */
    .top-header {
        background: linear-gradient(135deg, var(--medical-blue) 0%, var(--medical-teal) 100%);
        color: white;
        padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: var(--shadow);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .egypt-flag {
        width: 40px;
        height: 28px;
        border-radius: 4px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        display: flex;
        flex-direction: column;
    }
    
    .flag-red { background: var(--egypt-red); height: 33.33%; }
    .flag-white { background: white; height: 33.33%; }
    .flag-black { background: var(--egypt-black); height: 33.33%; }
    
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .header-right {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        font-size: 0.9rem;
    }
    
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        opacity: 0.9;
    }
    
    /* Main Container */
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Upload Section */
    .upload-section {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
        text-align: center;
        border: 2px dashed var(--medical-teal);
    }
    
    /* Analysis Section */
    .analysis-section {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
    }
    
    .section-title {
        color: var(--medical-blue);
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Language Toggle */
    .language-toggle {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
        justify-content: center;
    }
    
    .lang-btn {
        padding: 0.5rem 1rem;
        border-radius: 25px;
        border: 2px solid var(--medical-blue);
        background: white;
        color: var(--medical-blue);
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .lang-btn.active {
        background: var(--medical-blue);
        color: white;
    }
    
    /* Quick Questions Grid */
    .questions-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    
    .question-btn {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid var(--medical-blue);
        padding: 0.75rem 1rem;
        border-radius: var(--border-radius);
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: left;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .question-btn:hover {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }
    
    /* Results Section */
    .results-section {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 2rem 0;
        border-left: 4px solid var(--medical-green);
        box-shadow: var(--shadow);
    }
    
    .result-item {
        background: white;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: var(--border-radius);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 3px solid var(--medical-teal);
    }
    
    /* Translation Display */
    .translation-container {
        display: grid;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .translation-item {
        background: white;
        padding: 1.25rem;
        border-radius: var(--border-radius);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Language Badges */
    .lang-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .badge-en {
        background: var(--medical-blue);
        color: white;
    }
    
    .badge-ar {
        background: var(--medical-green);
        color: white;
    }
    
    /* Analyze Button */
    .analyze-btn {
        background: linear-gradient(135deg, var(--medical-blue) 0%, var(--medical-teal) 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: var(--border-radius);
        border: none;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .analyze-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(14, 165, 233, 0.3);
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Disclaimer */
    .disclaimer {
        background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%);
        border: 1px solid #fb923c;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-top: 2rem;
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .header-content {
            flex-direction: column;
            text-align: center;
        }
        
        .app-title {
            font-size: 1.5rem;
        }
        
        .main-container {
            padding: 0 0.5rem;
        }
        
        .upload-section, .analysis-section {
            padding: 1.5rem;
        }
        
        .questions-grid {
            grid-template-columns: 1fr;
        }
        
        .header-right {
            flex-direction: column;
            gap: 0.5rem;
        }
    }
    
    /* Utility Classes */
    .text-center { text-align: center; }
    .mb-1 { margin-bottom: 0.5rem; }
    .mb-2 { margin-bottom: 1rem; }
    .mt-2 { margin-top: 1rem; }
    .font-bold { font-weight: 700; }
    .text-lg { font-size: 1.125rem; }
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
    
    # Top Navigation Header
    st.markdown('''
    <div class="top-header">
        <div class="header-content">
            <div class="header-left">
                <div class="egypt-flag">
                    <div class="flag-red"></div>
                    <div class="flag-white"></div>
                    <div class="flag-black"></div>
                </div>
                <div>
                    <h1 class="app-title">🏥 Medical Vision AI</h1>
                    <p class="app-subtitle">Advanced Medical Image Analysis for Egypt</p>
                </div>
            </div>
            <div class="header-right">
                <div class="status-item">
                    <span>🛡️ Secure</span>
                </div>
                <div class="status-item">
                    <span>🌍 Arabic/English</span>
                </div>
                <div class="status-item">
                    <span>🤖 AI Powered</span>
                </div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Main Container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Load AI Models
    with st.spinner("🔄 Loading AI models..."):
        vqa_processor, vqa_model = load_medical_vqa_model()
    
    if vqa_processor and vqa_model:
        # Image Upload Section
        st.markdown('''
        <div class="upload-section">
            <h2 class="section-title">📤 Upload Medical Image</h2>
            <p>Upload X-rays, CT scans, MRI, or ultrasound images for AI analysis</p>
        </div>
        ''', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a medical image...", 
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Display uploaded image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Medical Image for Analysis", use_container_width=True)
                st.info(f"📊 Image: {image.size[0]}×{image.size[1]}px | Size: {round(uploaded_file.size/1024, 1)}KB")
            
            # Analysis Section
            st.markdown('''
            <div class="analysis-section">
                <h2 class="section-title">🔬 Medical Analysis</h2>
            </div>
            ''', unsafe_allow_html=True)
            
            # Language Selection
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🇺🇸 English", type="secondary" if st.session_state.lang == 'ar' else "primary"):
                    st.session_state.lang = 'en'
            with col2:
                if st.button("🇪🇬 العربية", type="secondary" if st.session_state.lang == 'en' else "primary"):
                    st.session_state.lang = 'ar'
            
            # Quick Questions
            st.markdown("**💭 Quick Questions:**")
            
            questions = {
                "en": [
                    "What abnormalities do you see in this image?",
                    "Are there any fractures or breaks visible?",
                    "Is this medical scan normal or abnormal?",
                    "Describe the key medical findings",
                    "Are there any signs of infection?",
                    "Is there evidence of a tumor or mass?",
                    "What is your diagnostic assessment?",
                    "Any signs of pneumonia or lung issues?"
                ],
                "ar": [
                    "ما هي التشوهات التي تراها في هذه الصورة؟",
                    "هل هناك أي كسور أو انقطاع مرئي؟",
                    "هل هذا الفحص الطبي طبيعي أم غير طبيعي؟",
                    "صف النتائج الطبية الرئيسية",
                    "هل هناك أي علامات للعدوى؟",
                    "هل هناك دليل على ورم أو كتلة؟",
                    "ما هو تقييمك التشخيصي؟",
                    "أي علامات للالتهاب الرئوي أو مشاكل في الرئة؟"
                ]
            }
            
            # Display questions in grid
            lang_questions = questions[st.session_state.lang]
            cols = st.columns(2)
            for i, question in enumerate(lang_questions):
                col = cols[i % 2]
                if col.button(question, key=f"q_{i}_{st.session_state.lang}"):
                    st.session_state.question = question
            
            # Custom Question Input
            placeholder = "Type your medical question here..." if st.session_state.lang == 'en' else "اكتب سؤالك الطبي هنا..."
            custom_question = st.text_area(
                "Your Custom Question:", 
                value=st.session_state.get('question', ''),
                placeholder=placeholder,
                height=100
            )
            
            # Analyze Button
            if st.button("🔬 Analyze Medical Image", type="primary", use_container_width=True):
                if not custom_question:
                    st.warning("⚠️ Please enter a question about the medical image")
                else:
                    # Process question and translation
                    question_is_arabic = is_arabic(custom_question)
                    
                    # Handle translation logic
                    if st.session_state.lang == 'en' and question_is_arabic:
                        display_question_en, _ = translate_text(custom_question, "ar", "en")
                        model_question = custom_question
                        display_question_ar = custom_question
                    elif st.session_state.lang == 'en' and not question_is_arabic:
                        model_question, _ = translate_text(custom_question, "en", "ar")
                        display_question_en = custom_question
                        display_question_ar = model_question
                    elif st.session_state.lang == 'ar' and not question_is_arabic:
                        model_question, _ = translate_text(custom_question, "en", "ar")
                        display_question_en = custom_question
                        display_question_ar = model_question
                    else:
                        display_question_en, _ = translate_text(custom_question, "ar", "en")
                        model_question = custom_question
                        display_question_ar = custom_question
                    
                    # Add medical context
                    contextualized_question = get_medical_context(model_question)
                    
                    # Analyze image
                    with st.spinner("🧠 Analyzing your medical image..."):
                        arabic_answer = analyze_medical_image(image, contextualized_question, vqa_processor, vqa_model)
                    
                    # Ensure answer is in Arabic
                    arabic_answer_display, _ = ensure_arabic_answer(arabic_answer)
                    
                    # Translate to English
                    with st.spinner("🌐 Translating results..."):
                        english_answer, _ = translate_text(arabic_answer, "ar", "en")
                    
                    # Display Results - Prominently
                    st.markdown('''
                    <div class="results-section">
                        <h2 class="section-title">🎯 Analysis Results</h2>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Question Display
                    st.markdown(f'''
                    <div class="translation-container">
                        <div class="translation-item">
                            <strong>❓ Your Question:</strong> {display_question_en}
                            <span class="lang-badge badge-en">EN</span>
                        </div>
                        <div class="translation-item rtl-text">
                            <strong>سؤالك:</strong> {display_question_ar}
                            <span class="lang-badge badge-ar">AR</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Answer Display
                    st.markdown(f'''
                    <div class="translation-container">
                        <div class="translation-item">
                            <strong>🔍 Medical Analysis:</strong> {english_answer}
                            <span class="lang-badge badge-en">EN</span>
                        </div>
                        <div class="translation-item rtl-text">
                            <strong>التحليل الطبي:</strong> {arabic_answer_display}
                            <span class="lang-badge badge-ar">AR</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Success message
                    st.success("✅ Analysis completed successfully!")
    
    else:
        st.error("❌ Failed to load AI models. Please refresh the page and try again.")
    
    # Medical Disclaimer
    st.markdown('''
    <div class="disclaimer">
        <h3>⚠️ Important Medical Disclaimer</h3>
        <p><strong>This AI tool is for educational and research purposes only.</strong></p>
        <ul>
            <li>🩺 Always consult qualified healthcare professionals for medical decisions</li>
            <li>🚫 This tool should NOT replace professional medical diagnosis</li>
            <li>🔬 AI analysis may contain errors and limitations</li>
            <li>🏥 Seek immediate medical attention for emergencies</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

if __name__ == "__main__":
    main()
