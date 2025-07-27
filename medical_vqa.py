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
# ... [CSS code remains unchanged] ...

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
