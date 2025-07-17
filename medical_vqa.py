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
    initial_sidebar_state="collapsed"
)

# Custom CSS with modern medical design and Egypt theme
st.markdown("""
<style>
    /* Remove default Streamlit styling */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 1200px;
    }
    
    /* Top Navigation Bar */
    .top-nav {
        background: linear-gradient(135deg, #0ea5e9 0%, #10b981 100%);
        padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 1rem 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .nav-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .nav-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .egypt-flag {
        width: 32px;
        height: 24px;
        border-radius: 4px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        position: relative;
    }
    
    .flag-red { height: 8px; background: #ce1126; }
    .flag-white { height: 8px; background: #ffffff; }
    .flag-black { height: 8px; background: #000000; }
    
    .app-title {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .nav-badges {
        display: flex;
        gap: 1rem;
        color: rgba(255,255,255,0.9);
        font-size: 0.9rem;
    }
    
    .nav-badge {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #f0f9ff 0%, #ecfdf5 100%);
        border-radius: 1rem;
        margin-bottom: 2rem;
        border: 1px solid #e0f2fe;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #0ea5e9 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #64748b;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Upload Section */
    .upload-card {
        background: white;
        border: 2px dashed #0ea5e9;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.1);
    }
    
    .upload-card:hover {
        border-color: #10b981;
        box-shadow: 0 8px 30px rgba(14, 165, 233, 0.2);
    }
    
    /* Results Section */
    .results-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%);
        border: 1px solid #0ea5e9;
        border-radius: 1rem;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 30px rgba(14, 165, 233, 0.15);
    }
    
    .result-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        color: #0ea5e9;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .analysis-result {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .translation-section {
        background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
        border: 1px solid #10b981;
        border-radius: 1rem;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
    }
    
    .translation-item {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 3px solid #10b981;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .language-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    
    .en-badge {
        background: #0ea5e9;
        color: white;
    }
    
    .ar-badge {
        background: #10b981;
        color: white;
    }
    
    /* Quick Features */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Medical Disclaimer */
    .disclaimer-card {
        background: linear-gradient(135deg, #fef3c7 0%, #ffffff 100%);
        border: 1px solid #f59e0b;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 2rem 0;
        border-left: 4px solid #f59e0b;
    }
    
    /* Suggested Questions */
    .question-btn {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #0ea5e9;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin: 0.25rem;
        display: inline-block;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #0ea5e9;
        font-weight: 500;
    }
    
    .question-btn:hover {
        background: #0ea5e9;
        color: white;
        transform: translateY(-2px);
    }
    
    /* Loading animations */
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #10b981 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        st.error(f"Translation Error: {str(e)}")
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
        return f"Error analyzing image: {str(e)}"

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
    
    # Top Navigation Bar
    st.markdown("""
    <div class="top-nav">
        <div class="nav-content">
            <div class="nav-left">
                <div class="egypt-flag">
                    <div class="flag-red"></div>
                    <div class="flag-white"></div>
                    <div class="flag-black"></div>
                </div>
                <h1 class="app-title">🏥 Medical Vision AI</h1>
            </div>
            <div class="nav-badges">
                <div class="nav-badge">🔒 Secure</div>
                <div class="nav-badge">🌐 Arabic/English</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h2 class="hero-title">AI-Powered Medical Image Analysis</h2>
        <p class="hero-subtitle">Upload medical images for instant AI analysis with bilingual support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🤖 Loading medical AI model..."):
        vqa_processor, vqa_model = load_medical_vqa_model()
    
    if not (vqa_processor and vqa_model):
        st.error("❌ Failed to load medical AI models. Please refresh the page.")
        return
    
    # Success message for model loading
    st.success("✅ Medical AI models loaded successfully!")
    
    # File upload section
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("### 📸 Upload Medical Image")
    uploaded_file = st.file_uploader(
        "Choose a medical image (X-ray, CT scan, MRI, ultrasound)...", 
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supported formats: JPG, PNG, BMP"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="📋 Uploaded Medical Image", use_container_width=True)
            st.info(f"📏 Image size: {image.size[0]}x{image.size[1]} pixels")
        
        with col2:
            # Language selector
            st.markdown("### 🌐 Select Language")
            lang = st.radio(
                "Choose your preferred language:",
                ["English", "العربية"],
                horizontal=True,
                index=0 if st.session_state.lang == 'en' else 1
            )
            st.session_state.lang = 'en' if lang == "English" else 'ar'
            
            # Question input section
            st.markdown("### ❓ Ask a Medical Question")
            
            # Suggested questions
            st.markdown("**💡 Suggested Questions:**")
            
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
            
            # Display suggested questions as clickable buttons
            cols = st.columns(2)
            for i, q in enumerate(questions[st.session_state.lang]):
                col = cols[i % 2]
                if col.button(q, key=f"q_{i}_{st.session_state.lang}", use_container_width=True):
                    st.session_state.question = q
            
            # Custom question input
            placeholder = "Type your question here..." if st.session_state.lang == 'en' else "اكتب سؤالك هنا..."
            question = st.text_area(
                "Your question:",
                value=st.session_state.get('question', ''),
                placeholder=placeholder,
                height=100
            )
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                if question:
                    # Store original question
                    original_question = question
                    question_is_arabic = is_arabic(question)
                    
                    # Prepare variables for display
                    display_question_en = ""
                    display_question_ar = ""
                    
                    # Handle question translation based on language and input
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
                    
                    else:  # Arabic UI, Arabic question
                        with st.spinner("🔄 Translating question to English..."):
                            display_question_en, success = translate_text(question, "ar", "en")
                            if not success:
                                display_question_en = question + " [Auto]"
                        model_question = question
                        display_question_ar = question
                    
                    # Add medical context
                    contextualized_question = get_medical_context(model_question)
                    
                    # Analyze image
                    with st.spinner("🤖 Analyzing medical image..."):
                        arabic_answer = analyze_medical_image(image, contextualized_question, vqa_processor, vqa_model)
                    
                    # Ensure answer is in Arabic
                    arabic_answer_display, arabic_translated = ensure_arabic_answer(arabic_answer)
                    
                    # Translate answer to English
                    with st.spinner("🔄 Translating answer to English..."):
                        english_answer, success = translate_text(arabic_answer, "ar", "en")
                        if not success:
                            english_answer = arabic_answer + " [Auto]"
                    
                    # Display Results Section
                    st.markdown("""
                    <div class="results-card">
                        <div class="result-header">
                            🔍 Analysis Results
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Question display
                    st.markdown("### 📝 Your Question")
                    st.markdown(f"""
                    <div class="translation-item">
                        <strong>English:</strong> {display_question_en} 
                        <span class="language-badge en-badge">EN</span>
                    </div>
                    <div class="translation-item rtl-text">
                        <strong>العربية:</strong> {display_question_ar} 
                        <span class="language-badge ar-badge">AR</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Answer display
                    st.markdown("### 🎯 AI Analysis Result")
                    st.markdown(f"""
                    <div class="analysis-result">
                        <div class="translation-item">
                            <strong>English Answer:</strong> {english_answer}
                            <span class="language-badge en-badge">EN</span>
                        </div>
                        <div class="translation-item rtl-text">
                            <strong>الإجابة العربية:</strong> {arabic_answer_display}
                            <span class="language-badge ar-badge">AR</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    st.warning("⚠️ Please enter a question about the medical image.")
    
    else:
        # Quick Features Section (when no image uploaded)
        st.markdown("""
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">📸</div>
                <h4>Quick Upload</h4>
                <p>Drag & drop or click to upload medical images</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <h4>AI Analysis</h4>
                <p>Advanced AI models for medical image classification</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <h4>Bilingual Support</h4>
                <p>Support for Arabic and English medical terms</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Medical Disclaimer
    st.markdown("""
    <div class="disclaimer-card">
        <h4>⚠️ Medical Disclaimer</h4>
        <p>This tool is for <strong>educational purposes only</strong>. Always consult qualified healthcare professionals for medical decisions, diagnosis, and treatment. AI responses may contain errors or limitations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("🇪🇬 **Made in Egypt** | 💡 **Medical AI Demonstration** | 🔬 **For Educational Use Only**")

if __name__ == "__main__":
    main()
