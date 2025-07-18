import streamlit as st
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import re
import time
from deep_translator import GoogleTranslator
from functools import lru_cache

# Configure page
st.set_page_config(
    page_title="MediVision AI - Smart Medical Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern E-Health Theme CSS Design with RTL support
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&display=swap');
    
    /* Root Variables - E-Health Theme */
    :root {
        --primary-blue: #1a73e8;      /* Deep professional blue */
        --primary-teal: #00bcd4;      /* Medical teal */
        --secondary-green: #34a853;   /* Health green */
        --accent-orange: #fbbc05;     /* Warm accent */
        --light-blue: #e8f0fe;        /* Light background blue */
        --success-green: #34a853;     /* Success green */
        --warning-yellow: #fbbc05;    /* Warning yellow */
        --error-red: #ea4335;         /* Error red */
        --dark-blue: #174ea6;         /* Dark blue */
        --light-gray: #f8f9fa;        /* Light gray */
        --medium-gray: #dadce0;       /* Medium gray */
        --dark-gray: #202124;         /* Dark text */
        --white: #ffffff;             /* White */
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    body {
        background-color: #f8fafc;
        color: var(--dark-gray);
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-green) 100%);
        color: white;
        padding: 1.8rem 1rem;
        text-align: center;
        margin-bottom: 2.5rem;
        border-radius: 0 0 1.8rem 1.8rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Navigation Tabs */
    .nav-tabs {
        display: flex;
        justify-content: center;
        gap: 1.2rem;
        margin-bottom: 2.2rem;
    }
    
    .nav-tab {
        padding: 0.8rem 1.8rem;
        border-radius: 2.2rem;
        background: white;
        color: var(--primary-blue);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        border: 2px solid var(--primary-blue);
        font-size: 1.05rem;
    }
    
    .nav-tab.active {
        background: var(--primary-blue);
        color: white;
        box-shadow: 0 4px 8px rgba(26, 115, 232, 0.3);
    }
    
    /* Main Content Container */
    .content-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1.2rem;
    }
    
    /* Card Styles */
    .card {
        background: var(--white);
        border-radius: 1.2rem;
        padding: 1.8rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.8rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-top: 4px solid var(--primary-blue);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    .card h3 {
        color: var(--primary-blue);
        margin-top: 0;
        margin-bottom: 1.5rem;
        font-size: 1.4rem;
        border-bottom: 2px solid var(--medium-gray);
        padding-bottom: 0.8rem;
    }
    
    /* Buttons */
    .btn {
        background: linear-gradient(to right, var(--primary-blue), var(--dark-blue));
        color: white;
        border: none;
        border-radius: 0.9rem;
        padding: 0.9rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(26, 115, 232, 0.3);
        cursor: pointer;
        font-size: 1rem;
        display: inline-block;
        text-align: center;
    }
    
    .btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(26, 115, 232, 0.4);
        background: linear-gradient(to right, var(--dark-blue), var(--primary-blue));
    }
    
    .btn-outline {
        background: transparent;
        color: var(--primary-blue);
        border: 2px solid var(--primary-blue);
    }
    
    .btn-outline:hover {
        background: var(--primary-blue);
        color: white;
    }
    
    /* Quick Questions */
    .question-btn {
        background: linear-gradient(to bottom right, var(--light-blue), #dbeafe);
        border: 1px solid var(--medium-gray);
        padding: 1rem;
        border-radius: 0.9rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.95rem;
        text-align: center;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--dark-blue);
        font-weight: 500;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.03);
        position: relative;
        overflow: hidden;
        width: 100%;
        margin-bottom: 0.9rem;
    }
    
    .question-btn:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 0;
        background: linear-gradient(to bottom right, var(--primary-blue), var(--secondary-green));
        opacity: 0;
        transition: all 0.3s ease;
        z-index: 0;
    }
    
    .question-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 14px rgba(0, 0, 0, 0.1);
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
        background: linear-gradient(to bottom right, #e8f5e9, #c8e6c9);
        padding: 1.8rem;
        border-radius: 0.9rem;
        border-left: 4px solid var(--success-green);
        margin: 1.8rem 0;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Translation Boxes */
    .translation-item {
        background: linear-gradient(to bottom right, #f8fafc, #f1f5f9);
        padding: 1.4rem;
        margin: 1.4rem 0;
        border-radius: 0.7rem;
        border-left: 4px solid var(--primary-blue);
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
    }
    
    .translation-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 18px rgba(0, 0, 0, 0.08);
    }
    
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.15rem;
        line-height: 1.7;
    }
    
    /* Language Badges */
    .language-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 0.9rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.7rem;
    }
    
    .english-badge {
        background: linear-gradient(to right, var(--primary-blue), var(--dark-blue));
        color: white;
    }
    
    .arabic-badge {
        background: linear-gradient(to right, var(--secondary-green), #0f9d58);
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
    .arabic-ui .card h3,
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
            font-size: 1.9rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .nav-tab {
            padding: 0.6rem 1.2rem;
            font-size: 0.95rem;
        }
        
        .content-container {
            padding: 0 0.8rem;
        }
        
        .main-columns {
            flex-direction: column;
        }
        
        .question-btn {
            font-size: 0.88rem;
            padding: 0.85rem 0.6rem;
        }
        
        .card {
            padding: 1.4rem;
        }
    }
    
    /* Two-column layout */
    .main-columns {
        display: flex;
        gap: 2rem;
        margin-top: 1.2rem;
    }
    
    .left-column {
        flex: 4;
    }
    
    .right-column {
        flex: 6;
    }
    
    /* Section Title */
    .section-title {
        font-size: 1.35rem;
        color: var(--primary-blue);
        margin-bottom: 1.4rem;
        padding-bottom: 0.7rem;
        border-bottom: 2px solid var(--primary-teal);
    }
    
    /* Streamlit Button Override */
    .stButton > button {
        background: linear-gradient(to right, var(--primary-blue), var(--dark-blue)) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.9rem !important;
        padding: 0.9rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 8px rgba(26, 115, 232, 0.3) !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 12px rgba(26, 115, 232, 0.4) !important;
        background: linear-gradient(to right, var(--dark-blue), var(--primary-blue)) !important;
    }
    
    /* File Uploader Styling */
    .stFileUploader > div > div {
        border: 2px dashed var(--medium-gray) !important;
        border-radius: 1rem !important;
        background: var(--light-gray) !important;
        padding: 2rem 1rem !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: var(--primary-blue) !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache models globally
@st.cache_resource(show_spinner=False)
def load_medical_vqa_model():
    try:
        model_name = "sharawy53/final_diploma_blip-med-rad-arabic"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"🚨 Error loading VQA model: {str(e)}")
        return None, None

# Cache translations for faster switching
@lru_cache(maxsize=1000)
def cached_translate_text(text, source_lang, target_lang):
    if not text.strip():
        return text, False
        
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated_text = translator.translate(text)
        return translated_text, True
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text, False

def is_arabic(text):
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def analyze_medical_image(image, question, processor, model):
    try:
        inputs = processor(image, question, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_length=100, num_beams=5)
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"🚨 Error analyzing image: {str(e)}"

def ensure_arabic_answer(answer):
    if is_arabic(answer):
        return answer, False
    try:
        translated, success = cached_translate_text(answer, 'en', 'ar')
        if success:
            return translated, True
        return answer, False
    except:
        return answer, False

def get_medical_context(question):
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

# Initialize session state
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'question' not in st.session_state:
    st.session_state.question = ''
if 'translation_cache' not in st.session_state:
    st.session_state.translation_cache = {}
if 'vqa_processor' not in st.session_state:
    st.session_state.vqa_processor = None
if 'vqa_model' not in st.session_state:
    st.session_state.vqa_model = None

def main():
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
        # Load models only once
        if st.session_state.vqa_processor is None or st.session_state.vqa_model is None:
            with st.spinner("🔄 Loading AI models..." if st.session_state.lang == 'en' else "🔄 جاري تحميل نماذج الذكاء الاصطناعي..."):
                processor, model = load_medical_vqa_model()
                st.session_state.vqa_processor = processor
                st.session_state.vqa_model = model
        else:
            processor = st.session_state.vqa_processor
            model = st.session_state.vqa_model
        
        if processor and model:
            # Create two main columns
            col1, col2 = st.columns([4, 6], gap="large")
            
            with col1:
                # Image Upload Section
                st.markdown(f'''
                <div class="card">
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
                    <div class="card">
                        <h3>{T["image_info_title"]}</h3>
                        <p><strong>{T["dimensions"]}:</strong> {image.size[0]} x {image.size[1]} pixels</p>
                        <p><strong>{T["size"]}:</strong> {round(uploaded_file.size / 1024, 1)} KB</p>
                        <p><strong>{T["format"]}:</strong> {uploaded_file.type}</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with col2:
                # Language Selection - Fast switching
                lang_col1, lang_col2 = st.columns([1, 1])
                with lang_col1:
                    if st.button("🇺🇸 English" if st.session_state.lang == 'ar' else "English", 
                               use_container_width=True,
                               type="primary" if st.session_state.lang == 'en' else "secondary"):
                        st.session_state.lang = 'en'
                with lang_col2:
                    if st.button("🇪🇬 العربية" if st.session_state.lang == 'en' else "العربية", 
                               use_container_width=True,
                               type="primary" if st.session_state.lang == 'ar' else "secondary"):
                        st.session_state.lang = 'ar'
                
                # Analysis Section
                st.markdown(f'''
                <div class="card">
                    <h3>{T["analysis_title"]}</h3>
                </div>
                ''', unsafe_allow_html=True)
                
                # Suggested Questions - Fast rendering
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
                    <strong style="font-size: 1.2rem; color: var(--primary-blue);">{T["suggested_questions"]}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # عرض الأسئلة في عمودين (4 صفوف)
                col_left, col_right = st.columns(2)
                
                with col_left:
                    for i in range(0, 8, 2):
                        q = questions[st.session_state.lang][i]
                        if st.button(q, key=f"q_left_{i}_{st.session_state.lang}", use_container_width=True):
                            st.session_state.question = q
                
                with col_right:
                    for i in range(1, 8, 2):
                        q = questions[st.session_state.lang][i]
                        if st.button(q, key=f"q_right_{i}_{st.session_state.lang}", use_container_width=True):
                            st.session_state.question = q

                # Custom Question
                placeholder = T["custom_question_placeholder"]
                question = st.text_area(
                    "Your Question:", 
                    value=st.session_state.get('question', ''),
                    placeholder=placeholder,
                    height=130,
                    label_visibility="collapsed"
                )
                
                # Analyze Button
                if st.button(T["analyze_button"], type="primary", use_container_width=True):
                    if uploaded_file is None:
                        st.warning("Please upload a medical image first" if st.session_state.lang == 'en' else "يرجى رفع صورة طبية أولاً")
                    elif not question:
                        st.warning("Please enter a question about the medical image" if st.session_state.lang == 'en' else "يرجى إدخال سؤال حول الصورة الطبية")
                    else:
                        # Translate question if needed (using cached translations)
                        question_is_arabic = is_arabic(question)
                        
                        if st.session_state.lang == 'en' and question_is_arabic:
                            display_question_en, _ = cached_translate_text(question, "ar", "en")
                            model_question = question
                            display_question_ar = question
                        elif st.session_state.lang == 'en' and not question_is_arabic:
                            model_question, _ = cached_translate_text(question, "en", "ar")
                            display_question_en = question
                            display_question_ar = model_question
                        elif st.session_state.lang == 'ar' and not question_is_arabic:
                            model_question, _ = cached_translate_text(question, "en", "ar")
                            display_question_en = question
                            display_question_ar = model_question
                        else:
                            display_question_en, _ = cached_translate_text(question, "ar", "en")
                            model_question = question
                            display_question_ar = question
                        
                        # Add medical context
                        contextualized_question = get_medical_context(model_question)
                        
                        # Analyze image
                        with st.spinner("🧠 Analyzing your medical image..." if st.session_state.lang == 'en' else "🧠 جاري تحليل صورتك الطبية..."):
                            arabic_answer = analyze_medical_image(image, contextualized_question, processor, model)
                        
                        # Ensure answer is in Arabic
                        arabic_answer_display, arabic_translated = ensure_arabic_answer(arabic_answer)
                        
                        # Translate to English
                        with st.spinner("🌐 Translating results..." if st.session_state.lang == 'en' else "🌐 جاري ترجمة النتائج..."):
                            english_answer, _ = cached_translate_text(arabic_answer, "ar", "en")
                        
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
        <div class="card">
            <h3>{T["about_title"]}</h3>
            <p>{T["about_content"]}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Features and Technology in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'''
            <div class="card">
                <h3>{T["features_title"]}</h3>
                <ul style="list-style-type: none; padding-left: 0; margin-top: 1rem;">
                    {''.join(f'<li style="margin-bottom: 1.2rem; padding-left: 1.5rem; position: relative;"><span style="position: absolute; left: 0; color: var(--primary-blue);">→</span>{feature}</li>' for feature in T["features"])}
                </ul>
            </div>
            ''', unsafe_allow_html=True)
            
        with col2:
            st.markdown(f'''
            <div class="card">
                <h3>{T["tech_title"]}</h3>
                <ul style="list-style-type: none; padding-left: 0; margin-top: 1rem;">
                    {''.join(f'<li style="margin-bottom: 1.2rem; padding-left: 1.5rem; position: relative;"><span style="position: absolute; left: 0; color: var(--primary-blue);">→</span>{tech}</li>' for tech in T["tech"])}
                </ul>
            </div>
            ''', unsafe_allow_html=True)
        
        # Medical disclaimer
        st.markdown(f'''
        <div class="card" style="background: linear-gradient(to bottom right, #fff8e1, #ffecb3); border-left: 4px solid var(--warning-yellow);">
            <h3>{T["professional_disclaimer"]}</h3>
            <p>{T["professional_content"]}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close content-container

if __name__ == "__main__":
    main()
