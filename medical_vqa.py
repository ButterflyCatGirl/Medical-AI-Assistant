import streamlit as st
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import re
import time
from deep_translator import GoogleTranslator
from functools import lru_cache

# Enhanced Medical Translation Dictionary (English to Arabic)
MEDICAL_TRANSLATION_DICT = {
    # General Terms
    "normal": "طبيعي",
    "abnormal": "غير طبيعي",
    "findings": "النتائج",
    "analysis": "تحليل",
    "diagnosis": "تشخيص",
    "impression": "انطباع",
    "observation": "ملاحظة",
    "image": "صورة",
    "kind": "نوع",
    "area": "منطقة",
    "medical": "طبي",
    "pleural": "جنبي",
    "efficacy": "فعالية",
    "paratracheal": "مجاور للرغامى",
    "mediastinal": "منصفية",
    "pulmonary": "رئوي",
    "paratracheal area": "منطقة مجاورة للرغامى",  # Added full phrase
    
    # Anatomy
    "lung": "رئة",
    "lungs": "رئتين",
    "heart": "قلب",
    "bone": "عظم",
    "bones": "عظام",
    "fracture": "كسر",
    "fractures": "كسور",
    "rib": "ضلع",
    "ribs": "أضلاع",
    "spine": "عمود فقري",
    "cavity": "تجويف",
    "tissue": "نسيج",
    
    # Conditions
    "pneumonia": "التهاب رئوي",
    "edema": "وذمة",
    "effusion": "انصباب",
    "consolidation": "تصلب",
    "opacity": "عتامة",
    "nodule": "عقدة",
    "nodules": "عقد",
    "mass": "كتلة",
    "tumor": "ورم",
    "infection": "عدوى",
    "inflammation": "التهاب",
    "degeneration": "تنكس",
    "cardiomegaly": "تضخم القلب",
    "pneumothorax": "استرواح الصدر",
    "atelectasis": "انخماص",
    
    # Descriptors
    "mild": "خفيف",
    "moderate": "متوسط",
    "severe": "شديد",
    "acute": "حاد",
    "chronic": "مزمن",
    "bilateral": "ثنائي الجانب",
    "unilateral": "أحادي الجانب",
    "diffuse": "منتشر",
    "focal": "بؤري",
    "enlarged": "متضخم",
    "enlargement": "تضخم",
    "calcification": "تكلس",
    "thickening": "سماكة",
    
    # Specific Medical Terms
    "pleural effusion": "انصباب جنبي",
    "pulmonary edema": "وذمة رئوية",
    "bone fracture": "كسر عظمي",
    "rib fracture": "كسر ضلعي",
    "lung infection": "عدوى رئوية",
    "heart enlargement": "تضخم القلب",
    "lung opacity": "عتامة رئوية",
    "lung consolidation": "تصلب رئوي",
    "pulmonary nodule": "عقدة رئوية",
    "mediastinal mass": "كتلة منصفية",
    "bone degeneration": "تنكس عظمي",
    "spinal degeneration": "تنكس فقري",
    
    # Imaging Types
    "x-ray": "أشعة سينية",
    "xray": "أشعة سينية",
    "ct scan": "تصوير مقطعي",
    "computed tomography": "تصوير مقطعي",
    "mri scan": "تصوير بالرنين المغناطيسي",
    "magnetic resonance": "تصوير بالرنين المغناطيسي",
    "ultrasound": "موجات فوق صوتية",
    "chest x-ray": "أشعة سينية على الصدر",
    "abdominal x-ray": "أشعة سينية على البطن",
    "bone x-ray": "أشعة سينية على العظام",
    "radiograph": "صورة إشعاعية",
    "mammogram": "تصوير الثدي الشعاعي",
    "imaging": "تصوير",
    "scan": "فحص",
}

# Location Terms Dictionary for anatomical positions
LOCATION_TERMS = {
    "right": "يمين",
    "left": "يسار",
    "upper": "علوي",
    "lower": "سفلي",
    "anterior": "أمامي",
    "posterior": "خلفي",
    "lateral": "جانبي",
    "medial": "إنسي",
    "proximal": "قريب",
    "distal": "بعيد",
    "superior": "علوي",
    "inferior": "سفلي",
    "central": "مركزي",
    "peripheral": "محيطي",
    "right side": "الجانب الأيمن",
    "left side": "الجانب الأيسر",
    "top": "أعلى",
    "bottom": "أسفل",
    "middle": "وسط",
    "side": "جانب"
}

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
    
    /* Quick Questions - UPDATED COLORS */
    .question-btn {
        background: linear-gradient(to bottom right, #e0f0ff, #d1f2eb); /* Light Nile blue + light green */
        border: 1px solid #b8e0d2; /* Soft green border */
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
        color: #1a6e8a; /* Deep blue text */
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
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-green) 100%); /* Same as header */
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
        model_name = "sharawy53/final_diploma_V3_blip-med-rad-arabic"
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
        # Apply medical dictionary translation first
        if source_lang == 'en' and target_lang == 'ar':
            for eng, ar in MEDICAL_TRANSLATION_DICT.items():
                if eng.lower() in text.lower():
                    return text.replace(eng, ar), True
        elif source_lang == 'ar' and target_lang == 'en':
            for eng, ar in MEDICAL_TRANSLATION_DICT.items():
                if ar in text:
                    return text.replace(ar, eng), True
        
        # Fallback to Google Translate
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated_text = translator.translate(text)
        return translated_text, True
    except Exception as e:
        return text, False

def is_valid_translation(text, target_lang):
    """Check if translation is valid"""
    if len(text) == 0:
        return False
        
    # Check for common error patterns
    error_patterns = [
        r'[A-Za-z]{15,}',  # Long sequences of Latin letters
        r'\b\w{1}\b',       # Single-letter words
        r'\d{5,}',          # Long digit sequences
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, text):
            return False
    
    # Check character ratio based on target language
    if target_lang == 'ar':
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        return arabic_chars / len(text) > 0.5 if text else False
    else:
        latin_chars = len(re.findall(r'[A-Za-z]', text))
        return latin_chars / len(text) > 0.5 if text else False

def ensure_translation_quality(text, source_lang, target_lang, max_retries=2):
    """Ensure translation quality with retry mechanism"""
    original_text = text
    for attempt in range(max_retries):
        translated, success = cached_translate_text(text, source_lang, target_lang)
        if success and is_valid_translation(translated, target_lang):
            return translated
        time.sleep(0.3)  # Brief delay before retry
    
    # Fallback: return original text if translation fails
    return original_text

def is_arabic(text):
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def analyze_medical_image(image, question, processor, model):
    try:
        inputs = processor(image, question, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_length=100, num_beams=5)
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        # Post-process answer for specific question types
        return post_process_answer(question, answer)
    except Exception as e:
        return f"🚨 Error analyzing image: {str(e)}"

def post_process_answer(question, answer):
    """Refine answers for specific question types"""
    question_lower = question.lower()
    
    # 1. Normal/Abnormal questions - SINGLE WORD ANSWER
    normal_keywords = ["normal", "abnormal", "طبيعي", "غير طبيعي", "طبيعية", "غير طبيعية"]
    if any(keyword in question_lower for keyword in normal_keywords):
        if "abnormal" in answer.lower() or "غير طبيعي" in answer or "غير طبيعية" in answer:
            return "Abnormal" if "en" in question_lower else "غير طبيعي"
        elif "normal" in answer.lower() or "طبيعي" in answer or "طبيعية" in answer:
            return "Normal" if "en" in question_lower else "طبيعي"
        
        # Infer from context
        abnormal_indicators = [
            "fracture", "pneumonia", "tumor", "infection", "effusion", 
            "opacity", "consolidation", "edema", "mass", "nodule",
            "كسر", "ورم", "عدوى", "انصباب", "عتامة", "تصلب", "وذمة", "كتلة", "عقدة"
        ]
        
        if any(indicator in answer.lower() for indicator in abnormal_indicators):
            return "Abnormal" if "en" in question_lower else "غير طبيعي"
        else:
            return "Normal" if "en" in question_lower else "طبيعي"
    
    # 2. Image type questions - SHORT ANSWER
    type_keywords = ["what type", "what kind", "نوع", "نوع الصورة", "نوع الأشعة", "أي نوع"]
    if any(keyword in question_lower for keyword in type_keywords):
        if "x-ray" in answer.lower() or "xray" in answer.lower() or "أشعة" in answer:
            return "X-ray" if "en" in question_lower else "أشعة سينية"
        elif "ct" in answer.lower() or "computed tomography" in answer.lower() or "مقطعي" in answer:
            return "CT scan" if "en" in question_lower else "تصوير مقطعي"
        elif "mri" in answer.lower() or "magnetic resonance" in answer.lower() or "رنين" in answer:
            return "MRI scan" if "en" in question_lower else "تصوير بالرنين المغناطيسي"
        elif "ultrasound" in answer.lower() or "موجات" in answer:
            return "Ultrasound" if "en" in question_lower else "موجات فوق صوتية"
        elif "chest" in question_lower and ("x-ray" in question_lower or "أشعة" in question_lower):
            return "Chest X-ray" if "en" in question_lower else "أشعة سينية على الصدر"
        elif "abdominal" in question_lower and ("x-ray" in question_lower or "أشعة" in question_lower):
            return "Abdominal X-ray" if "en" in question_lower else "أشعة سينية على البطن"
        else:
            return "Radiograph" if "en" in question_lower else "صورة إشعاعية"
    
    # 3. Location questions - IMPROVED TRANSLATION
    location_keywords = ["where", "أين", "location", "موقع", "region", "منطقة"]
    if any(keyword in question_lower for keyword in location_keywords):
        # Apply location terms translation
        for eng, ar in LOCATION_TERMS.items():
            if re.search(rf'\b{re.escape(eng)}\b', answer, re.IGNORECASE):
                answer = re.sub(rf'\b{re.escape(eng)}\b', ar, answer, flags=re.IGNORECASE)
        
        # Apply medical translation to anatomical terms
        for eng, ar in MEDICAL_TRANSLATION_DICT.items():
            if re.search(rf'\b{re.escape(eng)}\b', answer, re.IGNORECASE):
                answer = re.sub(rf'\b{re.escape(eng)}\b', ar, answer, flags=re.IGNORECASE)
        
        # Format location response
        if is_arabic(question):
            return "الموقع: " + answer
        else:
            return "Location: " + answer
    
    # 4. General medical terms translation
    return apply_medical_translation(answer)

def apply_medical_translation(answer):
    """Apply medical translation dictionary"""
    if is_arabic(answer):
        # Arabic to English
        for eng, ar in MEDICAL_TRANSLATION_DICT.items():
            answer = answer.replace(ar, eng)
    else:
        # English to Arabic
        for eng, ar in MEDICAL_TRANSLATION_DICT.items():
            pattern = r'\b' + re.escape(eng) + r'\b'
            answer = re.sub(pattern, ar, answer, flags=re.IGNORECASE)
    
    return answer

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
        "symptom": "medical symptom",
        "type": "medical imaging type",
        "kind": "medical imaging type",
        "نوع": "medical imaging type",
        "أي نوع": "medical imaging type"
    }
    
    for keyword, context in medical_keywords.items():
        if keyword.lower() in question.lower():
            return f"In the context of {context}: {question}"
    return question

# Enhanced bilingual texts with complete translations
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
    
    # Modern Header with gradient background
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
                # Image Upload Section with card design
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
                    st.image(image, caption=T["upload_title"], use_container_width=True)
                    
                    # Image info in card design
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
                
                # Analysis Section with card design
                st.markdown(f'''
                <div class="card">
                    <h3>{T["analysis_title"]}</h3>
                </div>
                ''', unsafe_allow_html=True)
                
                # Suggested Questions - Fast rendering with gradient buttons
                questions = {
                    "en": [
                        "Is this result normal or abnormal?",
                        "What kind of image is this?",
                        "Are there any fractures visible?",
                        "Describe the key medical findings",
                        "Any signs of infection present?",
                        "Is there a tumor or mass visible?",
                        "What is your diagnostic assessment?",
                        "Is there evidence of pneumonia?",
                        "Where is the abnormality located?",
                        "What is the most significant finding?"
                    ],
                    "ar": [
                        "هل هذه النتيجة طبيعية أم غير طبيعية؟",
                        "ما نوع هذه الصورة؟",
                        "هل هناك أي كسور مرئية؟",
                        "صف النتائج الطبية الرئيسية",
                        "هل هناك أي علامات للعدوى؟",
                        "هل هناك ورم أو كتلة مرئية؟",
                        "ما هو تقييمك التشخيصي؟",
                        "هل هناك دليل على الالتهاب الرئوي؟",
                        "أين يقع الخلل؟",
                        "ما هو الاكتشاف الأكثر أهمية؟"
                    ]
                }
                
                st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <strong style="font-size: 1.2rem; color: var(--primary-blue);">{T["suggested_questions"]}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Display questions in two columns with gradient buttons
                col_left, col_right = st.columns(2)
                
                with col_left:
                    for i in range(0, 10, 2):
                        q = questions[st.session_state.lang][i]
                        if st.button(q, key=f"q_left_{i}_{st.session_state.lang}", use_container_width=True):
                            st.session_state.question = q
                
                with col_right:
                    for i in range(1, 10, 2):
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
                
                # Analyze Button with gradient design
                if st.button(T["analyze_button"], type="primary", use_container_width=True):
                    if uploaded_file is None:
                        st.warning("Please upload a medical image first" if st.session_state.lang == 'en' else "يرجى رفع صورة طبية أولاً")
                    elif not question:
                        st.warning("Please enter a question about the medical image" if st.session_state.lang == 'en' else "يرجى إدخال سؤال حول الصورة الطبية")
                    else:
                        # تحديد لغة السؤال الأصلي
                        question_is_arabic = is_arabic(question)

                        # النموذج يحتاج سؤالاً بالعربية دائماً
                        if question_is_arabic:
                            model_question = question
                        else:
                            model_question = ensure_translation_quality(question, "en", "ar")

                        # تحضير السؤال للعرض:
                        if question_is_arabic:
                            display_question_ar = question
                            display_question_en = ensure_translation_quality(question, "ar", "en")
                        else:
                            display_question_en = question
                            display_question_ar = ensure_translation_quality(question, "en", "ar")
                        
                        # Add medical context
                        contextualized_question = get_medical_context(model_question)
                        
                        # Analyze image
                        with st.spinner("🧠 Analyzing your medical image..." if st.session_state.lang == 'en' else "🧠 جاري تحليل صورتك الطبية..."):
                            arabic_answer = analyze_medical_image(image, contextualized_question, processor, model)
                        
                        # Apply medical translation dictionary
                        arabic_answer = apply_medical_translation(arabic_answer)
                        
                        # Translate to English with quality check
                        with st.spinner("🌐 Translating results..." if st.session_state.lang == 'en' else "🌐 جاري ترجمة النتائج..."):
                            english_answer = ensure_translation_quality(arabic_answer, "ar", "en")
                        
                        # Display results in styled boxes
                        st.markdown(f'''
                        <div class="result-box">
                            <h3>{T["results_title"]}</h3>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Question display in translation boxes
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
                        
                        # Answer display in translation boxes
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
                            {arabic_answer}
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
        # About section with card design
        st.markdown(f'''
        <div class="card">
            <h3>{T["about_title"]}</h3>
            <p>{T["about_content"]}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Features and Technology in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Features section with gradient background
            st.markdown(f'''
            <div class="card">
                <h3>{T["features_title"]}</h3>
                <div style="padding: 1rem;">
                    <ul style="list-style-type: none; padding-left: 0;">
            ''', unsafe_allow_html=True)
            
            for feature in T["features"]:
                # Extract icon and text
                icon = feature.split(" ")[0]
                text = " ".join(feature.split(" ")[1:])
                st.markdown(f'''
                <li style="margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.8rem; background: linear-gradient(to right, #e0f7fa, #e1f5fe);">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                    <span style="font-size: 1.1rem;">{text}</span>
                </li>
                ''', unsafe_allow_html=True)
            
            st.markdown('''
                    </ul>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col2:
            # Technology section with gradient background
            st.markdown(f'''
            <div class="card">
                <h3>{T["tech_title"]}</h3>
                <div style="padding: 1rem;">
                    <ul style="list-style-type: none; padding-left: 0;">
            ''', unsafe_allow_html=True)
            
            for tech in T["tech"]:
                # Extract icon and text
                icon = tech.split(" ")[0]
                text = " ".join(tech.split(" ")[1:])
                st.markdown(f'''
                <li style="margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.8rem; background: linear-gradient(to right, #f3e5f5, #f1e6ff);">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                    <span style="font-size: 1.1rem;">{text}</span>
                </li>
                ''', unsafe_allow_html=True)
            
            st.markdown('''
                    </ul>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Medical disclaimer with yellow gradient
        st.markdown(f'''
        <div class="card" style="background: linear-gradient(to bottom right, #fff8e1, #ffecb3); border-left: 4px solid var(--warning-yellow);">
            <h3>{T["professional_disclaimer"]}</h3>
            <p>{T["professional_content"]}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close content-container

if __name__ == "__main__":
    main()
