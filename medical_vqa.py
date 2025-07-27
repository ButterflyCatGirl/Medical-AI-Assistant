import streamlit as st
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import re
import time
from deep_translator import GoogleTranslator
from functools import lru_cache

# Medical Translation Dictionary (English to Arabic)
MEDICAL_TRANSLATION_DICT = {
    # General Terms
    "normal": "طبيعي",
    "abnormal": "غير طبيعي",
    "findings": "النتائج",
    "analysis": "تحليل",
    "diagnosis": "تشخيص",
    "impression": "انطباع",
    "observation": "ملاحظة",
    
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
    "pleural": "جنبي",
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

def apply_medical_translation(answer):
    """Apply medical translation dictionary to improve accuracy"""
    # If answer is in English, translate using dictionary
    if not is_arabic(answer):
        for eng, ar in MEDICAL_TRANSLATION_DICT.items():
            # Case-insensitive replacement with word boundaries
            pattern = r'\b' + re.escape(eng) + r'\b'
            answer = re.sub(pattern, ar, answer, flags=re.IGNORECASE)
    return answer

def ensure_arabic_answer(answer):
    if is_arabic(answer):
        return answer, False
    
    # First apply medical dictionary translation
    medical_translated = apply_medical_translation(answer)
    
    # If medical translation changed the answer, return it
    if medical_translated != answer:
        return medical_translated, True
    
    # Fallback to Google Translate
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
    # ... [text dictionaries remain unchanged] ...
}

# Initialize session state
# ... [session state initialization remains unchanged] ...

def main():
    # ... [main function remains unchanged until analysis section] ...
    
    if active_tab == T["tab_analysis"]:
        # ... [model loading and UI code remains unchanged] ...
        
        if processor and model:
            # ... [UI columns setup remains unchanged] ...
            
            with col1:
                # ... [image upload code remains unchanged] ...
            
            with col2:
                # ... [language selection and UI code remains unchanged] ...
                
                # Analyze Button
                if st.button(T["analyze_button"], type="primary", use_container_width=True):
                    if uploaded_file is None:
                        st.warning("Please upload a medical image first" if st.session_state.lang == 'en' else "يرجى رفع صورة طبية أولاً")
                    elif not question:
                        st.warning("Please enter a question about the medical image" if st.session_state.lang == 'en' else "يرجى إدخال سؤال حول الصورة الطبية")
                    else:
                        # ... [question translation code remains unchanged] ...
                        
                        # Add medical context
                        contextualized_question = get_medical_context(model_question)
                        
                        # Analyze image
                        with st.spinner("🧠 Analyzing your medical image..." if st.session_state.lang == 'en' else "🧠 جاري تحليل صورتك الطبية..."):
                            arabic_answer = analyze_medical_image(image, contextualized_question, processor, model)
                        
                        # Apply medical translation dictionary
                        arabic_answer = apply_medical_translation(arabic_answer)
                        
                        # Ensure answer is in Arabic
                        arabic_answer_display, arabic_translated = ensure_arabic_answer(arabic_answer)
                        
                        # Translate to English
                        with st.spinner("🌐 Translating results..." if st.session_state.lang == 'en' else "🌐 جاري ترجمة النتائج..."):
                            english_answer, _ = cached_translate_text(arabic_answer_display, "ar", "en")
                        
                        # Display results
                        # ... [results display code remains unchanged] ...
    
    # ... [rest of the code remains unchanged] ...

if __name__ == "__main__":
    main()
