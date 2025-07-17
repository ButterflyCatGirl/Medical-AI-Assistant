import streamlit as st
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import re
import time
from deep_translator import GoogleTranslator

# Configure page
st.set_page_config(
    page_title="MedVision AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary: #2563eb;
        --secondary: #0ea5e9;
        --accent: #10b981;
        --light: #f0f9ff;
        --dark: #1e293b;
        --gray: #64748b;
        --success: #16a34a;
        --warning: #f59e0b;
        --error: #dc2626;
    }
    
    * {
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    
    .logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--dark);
    }
    
    .logo-icon {
        background-color: var(--primary);
        color: white;
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    
    .nav-tabs {
        display: flex;
        gap: 1.5rem;
        background: white;
        padding: 0.5rem;
        border-radius: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    
    .nav-tab {
        padding: 0.5rem 1.25rem;
        border-radius: 2rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-tab.active {
        background-color: var(--primary);
        color: white;
    }
    
    .nav-tab:hover:not(.active) {
        background-color: #f1f5f9;
    }
    
    .hero {
        text-align: center;
        padding: 3rem 0;
        margin-bottom: 2rem;
    }
    
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: var(--dark);
    }
    
    .hero p {
        font-size: 1.25rem;
        color: var(--gray);
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    .card {
        background: white;
        border-radius: 1rem;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: var(--dark);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-title .icon {
        background-color: var(--light);
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--primary);
    }
    
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 1rem;
        padding: 3rem 2rem;
        text-align: center;
        background-color: #f8fafc;
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .upload-area:hover {
        border-color: var(--primary);
        background-color: #f0f9ff;
    }
    
    .upload-icon {
        font-size: 3rem;
        color: var(--gray);
        margin-bottom: 1rem;
    }
    
    .image-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 2rem;
    }
    
    .preview-image {
        max-width: 100%;
        max-height: 400px;
        border-radius: 1rem;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .language-selector {
        display: flex;
        gap: 1rem;
        background: var(--light);
        padding: 0.75rem;
        border-radius: 2rem;
        margin-bottom: 1.5rem;
        justify-content: center;
    }
    
    .lang-btn {
        padding: 0.5rem 1.5rem;
        border-radius: 2rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        background: transparent;
        color: var(--dark);
    }
    
    .lang-btn.active {
        background-color: white;
        color: var(--primary);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    
    .suggested-questions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        justify-content: center;
    }
    
    .question-btn {
        background-color: var(--light);
        border: none;
        border-radius: 1rem;
        padding: 0.75rem 1.25rem;
        font-size: 0.95rem;
        color: var(--dark);
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .question-btn:hover {
        background-color: #e0f2fe;
        transform: translateY(-2px);
    }
    
    .input-area {
        margin-bottom: 1.5rem;
    }
    
    textarea {
        width: 100%;
        border-radius: 1rem;
        padding: 1.25rem;
        border: 1px solid #cbd5e1;
        font-size: 1rem;
        min-height: 120px;
        resize: vertical;
        transition: all 0.3s ease;
    }
    
    textarea:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    .analyze-btn {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 1rem;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
    }
    
    .analyze-btn:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.4);
    }
    
    .result-container {
        background: white;
        border-radius: 1rem;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
        padding: 2rem;
        margin-top: 2rem;
    }
    
    .result-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: var(--dark);
    }
    
    .result-card {
        background-color: #f8fafc;
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--accent);
    }
    
    .result-card-title {
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: var(--dark);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .result-content {
        color: var(--dark);
        line-height: 1.6;
    }
    
    .lang-tag {
        padding: 0.25rem 0.75rem;
        border-radius: 2rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .en-tag {
        background-color: #dbeafe;
        color: var(--primary);
    }
    
    .ar-tag {
        background-color: #dcfce7;
        color: var(--success);
    }
    
    .warning-tag {
        background-color: #fef3c7;
        color: var(--warning);
    }
    
    .disclaimer {
        background-color: #fffbeb;
        border-radius: 1rem;
        padding: 1.5rem;
        margin-top: 2rem;
        border-left: 4px solid var(--warning);
    }
    
    .disclaimer-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: var(--warning);
    }
    
    .about-content {
        line-height: 1.8;
        color: var(--dark);
    }
    
    .tech-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .tech-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .tech-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    
    .tech-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: var(--primary);
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        color: var(--gray);
        border-top: 1px solid #e2e8f0;
    }
    
    /* RTL support for Arabic */
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header {
            flex-direction: column;
            gap: 1rem;
        }
        
        .nav-tabs {
            width: 100%;
            justify-content: center;
        }
        
        .hero h1 {
            font-size: 2rem;
        }
        
        .hero p {
            font-size: 1.1rem;
        }
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
        st.session_state.lang = 'en'
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'analysis'
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header with navigation
    st.markdown("""
    <div class="header">
        <div class="logo">
            <div class="logo-icon">🩺</div>
            <span>MedVision AI</span>
        </div>
        <div class="nav-tabs">
            <div class="nav-tab %s" onclick="setActiveTab('analysis')">Image Analysis</div>
            <div class="nav-tab %s" onclick="setActiveTab('about')">About</div>
        </div>
    </div>
    """ % (
        "active" if st.session_state.active_tab == 'analysis' else "",
        "active" if st.session_state.active_tab == 'about' else ""
    ), unsafe_allow_html=True)
    
    # Add JS for tab switching
    st.markdown("""
    <script>
    function setActiveTab(tabName) {
        window.parent.document.querySelectorAll('div[role="tab"]').forEach(tab => {
            tab.classList.remove('active');
        });
        window.parent.document.querySelector(`div[onclick*="${tabName}"]`).classList.add('active');
        window.parent.postMessage({type: 'setActiveTab', tab: tabName}, '*');
    }
    
    window.addEventListener('message', function(event) {
        if (event.data.type === 'setActiveTab') {
            window.parent.document.querySelectorAll('div[role="tab"]').forEach(tab => {
                tab.classList.remove('active');
            });
            window.parent.document.querySelector(`div[onclick*="${event.data.tab}"]`).classList.add('active');
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Hero section
    if st.session_state.active_tab == 'analysis':
        st.markdown("""
        <div class="hero">
            <h1>Medical Image Analysis with AI</h1>
            <p>Upload medical images and get instant AI-powered insights. Ask questions in English or Arabic and receive detailed analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if st.session_state.active_tab == 'analysis':
        with st.container():
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                # Image upload section
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title"><div class="icon">🖼️</div> Upload Medical Image</div>', unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp"], 
                                               label_visibility="collapsed")
                
                if uploaded_file:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(image, use_column_width=True, caption="", output_format="PNG", 
                            use_container_width=True, clamp=True, 
                            channels="RGB", format="PNG", 
                            width=None)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="upload-area">
                        <div class="upload-icon">📁</div>
                        <h3>Drag & Drop your medical image here</h3>
                        <p>Supported formats: JPG, PNG, BMP</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close card
            
            with col2:
                # Analysis section
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title"><div class="icon">💬</div> Ask About the Image</div>', unsafe_allow_html=True)
                
                # Language selector
                st.markdown('<div class="language-selector">', unsafe_allow_html=True)
                if st.button("English", key="lang_en", type="primary" if st.session_state.lang == 'en' else "secondary"):
                    st.session_state.lang = 'en'
                if st.button("العربية", key="lang_ar", type="primary" if st.session_state.lang == 'ar' else "secondary"):
                    st.session_state.lang = 'ar'
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Suggested questions
                st.markdown('<div class="suggested-questions">', unsafe_allow_html=True)
                
                questions = {
                    "en": [
                        "What abnormalities do you see?",
                        "Are there any fractures?",
                        "Describe the key findings",
                        "Is there a tumor visible?",
                        "What is the diagnosis?"
                    ],
                    "ar": [
                        "ما هي التشوهات التي تراها؟",
                        "هل هناك أي كسور؟",
                        "صف النتائج الرئيسية",
                        "هل هناك ورم مرئي؟",
                        "ما هو التشخيص؟"
                    ]
                }
                
                for q in questions[st.session_state.lang]:
                    if st.button(q, key=f"q_{q[:5]}", use_container_width=True):
                        st.session_state.question = q
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Question input
                st.markdown('<div class="input-area">', unsafe_allow_html=True)
                placeholder = "Type your question here..." if st.session_state.lang == 'en' else "اكتب سؤالك هنا..."
                question = st.text_area("", value=st.session_state.get('question', ''), 
                                      placeholder=placeholder, height=120,
                                      label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Analyze button
                if st.button("🔍 Analyze Image", key="analyze_btn", use_container_width=True):
                    st.session_state.analyze_clicked = True
                else:
                    st.session_state.analyze_clicked = False
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close card
        
        # Load models only when needed
        if 'vqa_processor' not in st.session_state or 'vqa_model' not in st.session_state:
            with st.spinner("Loading AI models..."):
                vqa_processor, vqa_model = load_medical_vqa_model()
                st.session_state.vqa_processor = vqa_processor
                st.session_state.vqa_model = vqa_model
        
        # Display results if analyze was clicked
        if st.session_state.get('analyze_clicked', False) and uploaded_file and question:
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
                                                     st.session_state.vqa_processor, 
                                                     st.session_state.vqa_model)
            
            # Ensure the answer is properly in Arabic
            arabic_answer_display, arabic_translated = ensure_arabic_answer(arabic_answer)
            
            # Always translate the answer to English
            with st.spinner("Translating answer to English..."):
                english_answer, success = translate_text(
                    arabic_answer, "ar", "en"
                )
                if not success:
                    english_answer = arabic_answer + " [Auto]"
            
            # Display results
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown('<div class="result-title"><div>📋</div> Analysis Results</div>', unsafe_allow_html=True)
            
            # Question display
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-card-title">Your Question</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-content">{display_question_en} <span class="lang-tag en-tag">EN</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="result-card rtl-text">', unsafe_allow_html=True)
            st.markdown('<div class="result-card-title">سؤالك</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-content">{display_question_ar} <span class="lang-tag ar-tag">AR</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Answer display
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-card-title">Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-content">{english_answer} <span class="lang-tag en-tag">EN</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="result-card rtl-text">', unsafe_allow_html=True)
            st.markdown('<div class="result-card-title">الإجابة</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-content">{arabic_answer_display} <span class="lang-tag ar-tag">AR</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
                <div class="disclaimer-title">⚠️ Medical Disclaimer</div>
                <p>This analysis is for educational purposes only. Always consult healthcare professionals for medical decisions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close result-container
    
    elif st.session_state.active_tab == 'about':
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="icon">ℹ️</div> About MedVision AI</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="about-content">
            <p>MedVision AI is an advanced medical imaging analysis tool powered by artificial intelligence. 
            It helps medical professionals and students analyze medical images and get instant insights.</p>
            
            <h3>Key Features</h3>
            <ul>
                <li>AI-powered analysis of medical images (X-rays, CT scans, MRIs)</li>
                <li>Bilingual support for English and Arabic</li>
                <li>Intuitive question-answering interface</li>
                <li>Instant results with detailed explanations</li>
                <li>Educational tool for medical students</li>
            </ul>
            
            <h3>How It Works</h3>
            <ol>
                <li>Upload a medical image (X-ray, CT scan, MRI, etc.)</li>
                <li>Ask questions about the image in English or Arabic</li>
                <li>Receive AI-powered analysis with detailed findings</li>
                <li>View results in both languages for better understanding</li>
            </ol>
            
            <h3>Technologies Used</h3>
            <div class="tech-grid">
                <div class="tech-card">
                    <div class="tech-icon">🧠</div>
                    <h4>BLIP Model</h4>
                    <p>Medical VQA for image analysis</p>
                </div>
                <div class="tech-card">
                    <div class="tech-icon">🤖</div>
                    <h4>PyTorch</h4>
                    <p>Deep learning framework</p>
                </div>
                <div class="tech-card">
                    <div class="tech-icon">🌐</div>
                    <h4>Streamlit</h4>
                    <p>Web application interface</p>
                </div>
                <div class="tech-card">
                    <div class="tech-icon">🔤</div>
                    <h4>Translator</h4>
                    <p>Bilingual support</p>
                </div>
            </div>
            
            <h3>Disclaimer</h3>
            <p>This application is intended for educational and research purposes only. 
            It is not a substitute for professional medical advice, diagnosis, or treatment. 
            Always seek the advice of qualified healthcare providers with any questions you may have regarding medical conditions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close card
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2023 MedVision AI | Medical Image Analysis System</p>
        <p>For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

if __name__ == "__main__":
    main()
