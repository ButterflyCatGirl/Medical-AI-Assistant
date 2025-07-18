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

# Modern Medical Theme CSS Design with RTL support
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&display=swap');
    
    /* Root Variables - Medical Theme */
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
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    body {
        background-color: #f0f9ff;
        color: #1e293b;
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
        border-left: 4px solid var(--primary-blue);
    }
    
    /* Analysis Section */
    .analysis-section {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        border-left: 4px solid var(--primary-teal);
    }
    
    /* Quick Questions - 2 columns layout with medical theme */
    .quick-questions {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.8rem;
        margin-bottom: 1.5rem;
    }
    
    .question-btn {
        background: linear-gradient(to bottom right, #e0f2fe, #dbeafe);
        border: 1px solid #bae6fd;
        padding: 0.8rem;
        border-radius: 0.75rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.95rem;
        text-align: center;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #1e40af;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .question-btn:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 0;
        background: linear-gradient(to bottom right, var(--primary-blue), var(--primary-teal));
        opacity: 0;
        transition: all 0.3s ease;
        z-index: 0;
    }
    
    .question-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        color: white;
        border-color: var(--primary-blue);
    }
    
    .question-btn:hover:before {
        height: 100%;
        opacity: 1;
    }
    
    .question-btn span {
        position: relative;
        z-index: 1;
    }
    
    /* Result Boxes */
    .result-box {
        background: linear-gradient(to bottom right, #f0fdf4, #dcfce7);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid var(--success-green);
        margin: 1.5rem 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Translation Boxes */
    .translation-item {
        background: linear-gradient(to bottom right, #f8fafc, #f1f5f9);
        padding: 1.2rem;
        margin: 1.2rem 0;
        border-radius: 0.5rem;
        border-left: 3px solid var(--primary-blue);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
    }
    
    .translation-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }
    
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.1rem;
    }
    
    /* Language Badges */
    .language-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .english-badge {
        background: linear-gradient(to right, var(--primary-blue), var(--dark-blue));
        color: white;
    }
    
    .arabic-badge {
        background: linear-gradient(to right, var(--primary-teal), var(--success-green));
        color: white;
    }
    
    /* Arabic UI Elements */
    .arabic-ui .rtl-text,
    .arabic-ui .section-title,
    .arabic-ui .translation-item,
    .arabic-ui .question-btn,
    .arabic-ui .nav-tab,
    .arabic-ui .main-header h1,
    .arabic-ui .main-header p,
    .arabic-ui .image-section h3,
    .arabic-ui .analysis-section h3,
    .arabic-ui .result-box h3,
    .arabic-ui .translation-item strong,
    .arabic-ui .language-badge {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .arabic-ui .nav-tabs {
        direction: rtl;
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
        
        .main-columns {
            flex-direction: column;
        }
        
        .quick-questions {
            grid-template-columns: 1fr;
        }
    }
    
    /* Two-column layout */
    .main-columns {
        display: flex;
        gap: 1.8rem;
        margin-top: 1rem;
    }
    
    .left-column {
        flex: 4;
    }
    
    .right-column {
        flex: 6;
    }
    
    /* Section Title */
    .section-title {
        font-size: 1.3rem;
        color: var(--primary-blue);
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid var(--primary-teal);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(to right, var(--primary-blue), var(--primary-teal));
        color: white;
        border: none;
        border-radius: 0.8rem;
        padding: 0.8rem 1.8rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(14, 165, 233, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(14, 165, 233, 0.4);
        background: linear-gradient(to right, var(--dark-blue), var(--primary-blue));
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

# النصوص باللغتين
texts = {
    "en": {
        "app_title": "🩺 MediVision AI",
        "app_subtitle": "Advanced Medical Image Analysis & Multilingual Support",
        "tab_analysis": "🔬 Medical Analysis",
        "tab_about": "ℹ️ About",
        "upload_title": "📤 Upload Medical Image",
        "upload_prompt": "Choose a medical image...",
        "image_info_title": "📊 Image Information",
        "dimensions": "📏 Dimensions",
        "size": "📁 Size",
        "format": "🎨 Format",
        "analysis_title": "❓ Ask Medical Questions",
        "suggested_questions": "Suggested Questions:",
        "custom_question_placeholder": "Type your medical question here...",
        "analyze_button": "🔬 Analyze Medical Image",
        "results_title": "🔍 Medical Analysis Results",
        "question_label": "Question",
        "analysis_label": "Analysis",
        "disclaimer_title": "⚠️ Medical Disclaimer",
        "disclaimer_content": "This AI analysis is for educational purposes only. Always consult with qualified healthcare professionals for medical decisions. AI responses may contain errors and should not replace professional medical judgment.",
        "about_title": "ℹ️ About MediVision AI",
        "about_content": "Advanced medical image analysis platform combining cutting-edge AI technologies with multilingual support for healthcare professionals and medical students worldwide.",
        "features_title": "🔍 Core Features",
        "features": [
            "🩻 X-ray, CT, MRI & Ultrasound analysis",
            "🌍 English/Arabic bilingual support",
            "🧠 Specialized medical AI models",
            "🎯 Context-aware understanding",
            "💬 Natural language interaction",
            "📊 Detailed medical insights"
        ],
        "tech_title": "🛠️ Technology",
        "tech": [
            "🤖 BLIP Vision-Language Model",
            "🔥 PyTorch Deep Learning",
            "🌐 Google Translator API",
            "🚀 Streamlit Framework",
            "🐍 Python Backend",
            "💾 Hugging Face Transformers"
        ],
        "professional_disclaimer": "🩺 Professional Medical Disclaimer",
        "professional_content": "This is a demonstration application for educational and research purposes only. Always consult with qualified healthcare professionals for medical decisions, diagnosis, and treatment. AI-generated analysis should never replace professional medical judgment."
    },
    "ar": {
        "app_title": "🩺 رؤية طبية AI",
        "app_subtitle": "تحليل متقدم للصور الطبية مع دعم متعدد اللغات",
        "tab_analysis": "🔬 التحليل الطبي",
        "tab_about": "ℹ️ حول التطبيق",
        "upload_title": "📤 رفع صورة طبية",
        "upload_prompt": "اختر صورة طبية...",
        "image_info_title": "📊 معلومات الصورة",
        "dimensions": "📏 الأبعاد",
        "size": "📁 الحجم",
        "format": "🎨 الصيغة",
        "analysis_title": "❓ اطرح أسئلة طبية",
        "suggested_questions": "أسئلة مقترحة:",
        "custom_question_placeholder": "اكتب سؤالك الطبي هنا...",
        "analyze_button": "🔬 تحليل الصورة الطبية",
        "results_title": "🔍 نتائج التحليل الطبي",
        "question_label": "السؤال",
        "analysis_label": "التحليل",
        "disclaimer_title": "⚠️ تنبيه طبي",
        "disclaimer_content": "هذا التحليل بالذكاء الاصطناعي لأغراض تعليمية فقط. استشر دائمًا متخصصي الرعاية الصحية المؤهلين لاتخاذ القرارات الطبية. قد تحتوي استجابات الذكاء الاصطناعي على أخطاء ولا ينبغي أن تحل محل الحكم الطبي المهني.",
        "about_title": "ℹ️ حول تطبيق رؤية طبية AI",
        "about_content": "منصة تحليل الصور الطبية المتقدمة التي تجمع بين أحدث تقنيات الذكاء الاصطناعي مع الدعم متعدد اللغات لمتخصصي الرعاية الصحية وطلاب الطب في جميع أنحاء العالم.",
        "features_title": "🔍 الميزات الأساسية",
        "features": [
            "🩻 تحليل صور الأشعة السينية، التصوير المقطعي، الرنين المغناطيسي والموجات فوق الصوتية",
            "🌍 دعم ثنائي اللغة (الإنجليزية/العربية)",
            "🧠 نماذج ذكاء اصطناعي طبية متخصصة",
            "🎯 فهم واعٍ بالسياق",
            "💬 تفاعل بلغة طبيعية",
            "📊 رؤى طبية مفصلة"
        ],
        "tech_title": "🛠️ التقنية",
        "tech": [
            "🤖 نموذج BLIP للرؤية واللغة",
            "🔥 تعلم عميق باستخدام PyTorch",
            "🌐 واجهة برمجة تطبيقات الترجمة من جوجل",
            "🚀 إطار عمل Streamlit",
            "🐍 بايثون في الخلفية",
            "💾 Hugging Face Transformers"
        ],
        "professional_disclaimer": "🩺 تنبيه طبي احترافي",
        "professional_content": "هذا تطبيق توضيحي لأغراض تعليمية وبحثية فقط. استشر دائمًا متخصصي الرعاية الصحية المؤهلين لاتخاذ القرارات الطبية والتشخيص والعلاج. لا ينبغي أبدًا أن يحل التحليل الذي يولده الذكاء الاصطناعي محل الحكم الطبي المهني."
    }
}

def main():
    # Initialize session state
    if 'question' not in st.session_state:
        st.session_state.question = ''
    if 'lang' not in st.session_state:
        st.session_state.lang = 'en'
    
    # اختصار للنصوص
    T = texts[st.session_state.lang]
    
    # تحديد الفئة حسب اللغة
    ui_class = "arabic-ui" if st.session_state.lang == 'ar' else ""
    
    # Modern Header
    st.markdown(f'''
    <div class="main-header">
        <h1>{T["app_title"]}</h1>
        <p>{T["app_subtitle"]}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Navigation Tabs
    tabs = [T["tab_analysis"], T["tab_about"]]
    active_tab = st.radio(
        "Navigation:", 
        tabs, 
        horizontal=True, 
        label_visibility="collapsed",
        index=0
    )
    
    # Main content container
    st.markdown(f'<div class="content-container {ui_class}">', unsafe_allow_html=True)
    
    if active_tab == T["tab_analysis"]:
        # Load models
        with st.spinner("🔄 Loading AI models..."):
            vqa_processor, vqa_model = load_medical_vqa_model()
        
        if vqa_processor and vqa_model:
            # Create two main columns
            col1, col2 = st.columns([4, 6], gap="large")
            
            with col1:
                # Image Upload Section
                st.markdown(f'''
                <div class="image-section">
                    <h3>{T["upload_title"]}</h3>
                </div>
                ''', unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader(
                    T["upload_prompt"], 
                    type=["jpg", "jpeg", "png", "bmp"],
                    label_visibility="collapsed"
                )
                
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Medical Image for Analysis", use_container_width=True)
                    
                    # Image info
                    st.markdown(f'''
                    <div class="image-section">
                        <h3>{T["image_info_title"]}</h3>
                        <p><strong>{T["dimensions"]}:</strong> {image.size[0]} x {image.size[1]} pixels</p>
                        <p><strong>{T["size"]}:</strong> {round(uploaded_file.size / 1024, 1)} KB</p>
                        <p><strong>{T["format"]}:</strong> {uploaded_file.type}</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with col2:
                # Analysis Section
                st.markdown(f'''
                <div class="analysis-section">
                    <h3>{T["analysis_title"]}</h3>
                </div>
                ''', unsafe_allow_html=True)
                
                # Language Selection
                lang = st.radio(
                    "Language:", 
                    ["🇺🇸 English", "🇪🇬 العربية"],
                    horizontal=True,
                    index=0 if st.session_state.lang == 'en' else 1,
                    key="lang_selector"
                )
                
                st.session_state.lang = 'en' if lang == "🇺🇸 English" else 'ar'
                
                # Suggested Questions - Now in 2 columns with new style
                questions = {
                    "en": [
                        "What abnormalities do you see?",
                        "Are there any fractures visible?",
                        "Is this result normal or abnormal?",
                        "Describe the key medical findings",
                        "Any signs of infection present?",
                        "Is there a tumor or mass visible?",
                        "What is your diagnostic assessment?",
                        "Is there evidence of pneumonia?"
                    ],
                    "ar": [
                        "ما هي التشوهات التي تراها؟",
                        "هل هناك أي كسور مرئية؟",
                        "هل هذه النتيجة طبيعية أم غير طبيعية؟",
                        "صف النتائج الطبية الرئيسية",
                        "هل هناك أي علامات للعدوى؟",
                        "هل هناك ورم أو كتلة مرئية؟",
                        "ما هو تقييمك التشخيصي؟",
                        "هل هناك دليل على الالتهاب الرئوي؟"
                    ]
                }
                
                st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <strong style="font-size: 1.1rem; color: #1e40af;">{T["suggested_questions"]}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a grid container for questions
                st.markdown('<div class="quick-questions">', unsafe_allow_html=True)
                
                # Display questions in 2 columns
                lang_questions = questions[st.session_state.lang]
                for i, q in enumerate(lang_questions):
                    if st.button(q, key=f"q_{i}_{st.session_state.lang}"):
                        st.session_state.question = q
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Custom Question
                placeholder = T["custom_question_placeholder"]
                question = st.text_area(
                    "Your Question:", 
                    value=st.session_state.get('question', ''),
                    placeholder=placeholder,
                    height=120,
                    label_visibility="collapsed"
                )
                
                # Analyze Button
                if st.button(T["analyze_button"], type="primary", use_container_width=True):
                    if uploaded_file is None:
                        st.warning("Please upload a medical image first" if st.session_state.lang == 'en' else "يرجى رفع صورة طبية أولاً")
                    elif not question:
                        st.warning("Please enter a question about the medical image" if st.session_state.lang == 'en' else "يرجى إدخال سؤال حول الصورة الطبية")
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
                        with st.spinner("🧠 Analyzing your medical image..." if st.session_state.lang == 'en' else "🧠 جاري تحليل صورتك الطبية..."):
                            arabic_answer = analyze_medical_image(image, contextualized_question, vqa_processor, vqa_model)
                        
                        # Ensure answer is in Arabic
                        arabic_answer_display, arabic_translated = ensure_arabic_answer(arabic_answer)
                        
                        # Translate to English
                        with st.spinner("🌐 Translating results..." if st.session_state.lang == 'en' else "🌐 جاري ترجمة النتائج..."):
                            english_answer, _ = translate_text(arabic_answer, "ar", "en")
                        
                        # Display results
                        st.markdown(f'''
                        <div class="result-box">
                            <h3>{T["results_title"]}</h3>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Question display
                        st.markdown(f'''
                        <div class="translation-item">
                            <strong>{T["question_label"]}:</strong> 
                            {display_question_en}
                            <span class="language-badge english-badge">EN</span>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        st.markdown(f'''
                        <div class="translation-item rtl-text">
                            <strong>{T["question_label"]}:</strong> 
                            {display_question_ar}
                            <span class="language-badge arabic-badge">AR</span>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Answer display
                        st.markdown(f'''
                        <div class="translation-item">
                            <strong>{T["analysis_label"]}:</strong> 
                            {english_answer}
                            <span class="language-badge english-badge">EN</span>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        st.markdown(f'''
                        <div class="translation-item rtl-text">
                            <strong>{T["analysis_label"]}:</strong> 
                            {arabic_answer_display}
                            <span class="language-badge arabic-badge">AR</span>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Medical disclaimer
                        st.info(f"""
                        **{T["disclaimer_title"]}**  
                        {T["disclaimer_content"]}
                        """)
        
        else:
            st.error("Failed to load AI models. Please try again later." if st.session_state.lang == 'en' else "فشل تحميل نماذج الذكاء الاصطناعي. يرجى المحاولة لاحقًا.")
    
    elif active_tab == T["tab_about"]:
        # About section
        st.markdown(f'''
        <div class="analysis-section">
            <h3>{T["about_title"]}</h3>
            <p>{T["about_content"]}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Features and Technology in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(T["features_title"])
            st.markdown(f"""
            <div style="background: linear-gradient(to bottom right, #e0f2fe, #dbeafe); 
                        padding: 1.2rem; border-radius: 0.8rem; margin-bottom: 1.5rem;">
                <ul style="list-style-type: none; padding-left: 0;">
                    {''.join(f'<li style="margin-bottom: 0.8rem;">{feature}</li>' for feature in T["features"])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.subheader(T["tech_title"])
            st.markdown(f"""
            <div style="background: linear-gradient(to bottom right, #ede9fe, #e0e7ff); 
                        padding: 1.2rem; border-radius: 0.8rem; margin-bottom: 1.5rem;">
                <ul style="list-style-type: none; padding-left: 0;">
                    {''.join(f'<li style="margin-bottom: 0.8rem;">{tech}</li>' for tech in T["tech"])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Medical disclaimer
        st.info(f"""
        **{T["professional_disclaimer"]}**  
        {T["professional_content"]}
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close content-container

if __name__ == "__main__":
    main()
