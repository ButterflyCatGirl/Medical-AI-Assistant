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
    "paratracheal": "مجاور للرغامى",
    "mediastinal": "منصفية",
    "pulmonary": "رئوي",
    
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
    
    # Technology Terms
    "vision-language model": "نموذج الرؤية واللغة",
    "deep learning": "تعلم عميق",
    "translator api": "واجهة برمجة تطبيقات الترجمة",
    "framework": "إطار عمل",
    "backend": "خلفية",
    "transformers": "المحولات",
    "artificial intelligence": "الذكاء الاصطناعي",
    "medical ai": "الذكاء الاصطناعي الطبي",
    "context-aware": "واعي بالسياق",
    "natural language": "لغة طبيعية",
    "detailed insights": "رؤى مفصلة",
}

# Configure page
st.set_page_config(
    page_title="MediVision AI - Smart Medical Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern E-Health Theme CSS Design with RTL support
# ... [CSS code remains the same] ...

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
        # First try medical dictionary translation
        if source_lang == 'en' and target_lang == 'ar':
            for eng, ar in MEDICAL_TRANSLATION_DICT.items():
                if eng in text.lower():
                    text = text.replace(eng, ar)
                    return text, True
        
        # Fallback to Google Translate
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
        
        # Post-process answer for specific question types
        return post_process_answer(question, answer)
    except Exception as e:
        return f"🚨 Error analyzing image: {str(e)}"

def post_process_answer(question, answer):
    """Refine answers for specific question types to be more direct and accurate"""
    # Normal/Abnormal questions
    normal_keywords = ["normal", "abnormal", "طبيعي", "غير طبيعي", "طبيعية", "غير طبيعية"]
    if any(keyword in question.lower() for keyword in normal_keywords):
        # Check for explicit answers first
        if "abnormal" in answer.lower() or "غير طبيعي" in answer or "غير طبيعية" in answer:
            return "Abnormal" if "en" in question.lower() else "غير طبيعي"
        elif "normal" in answer.lower() or "طبيعي" in answer or "طبيعية" in answer:
            return "Normal" if "en" in question.lower() else "طبيعي"
        
        # Infer from context for more accurate determination
        abnormal_indicators = [
            "fracture", "pneumonia", "tumor", "infection", "effusion", 
            "opacity", "consolidation", "edema", "mass", "nodule",
            "كسر", "ورم", "عدوى", "انصباب", "عتامة", "تصلب", "وذمة", "كتلة", "عقدة"
        ]
        
        if any(indicator in answer.lower() for indicator in abnormal_indicators):
            return "Abnormal" if "en" in question.lower() else "غير طبيعي"
        else:
            return "Normal" if "en" in question.lower() else "طبيعي"
    
    # Image type questions
    type_keywords = ["what type", "what kind", "نوع", "نوع الصورة", "نوع الأشعة", "أي نوع"]
    if any(keyword in question.lower() for keyword in type_keywords):
        # Try to extract a direct answer
        if "x-ray" in answer.lower() or "xray" in answer.lower() or "أشعة" in answer:
            return "X-ray" if "en" in question.lower() else "أشعة سينية"
        elif "ct" in answer.lower() or "computed tomography" in answer.lower() or "مقطعي" in answer:
            return "CT scan" if "en" in question.lower() else "تصوير مقطعي"
        elif "mri" in answer.lower() or "magnetic resonance" in answer.lower() or "رنين" in answer:
            return "MRI scan" if "en" in question.lower() else "تصوير بالرنين المغناطيسي"
        elif "ultrasound" in answer.lower() or "موجات" in answer:
            return "Ultrasound" if "en" in question.lower() else "موجات فوق صوتية"
        elif "chest" in question.lower() and ("x-ray" in question.lower() or "أشعة" in question):
            return "Chest X-ray" if "en" in question.lower() else "أشعة سينية على الصدر"
        elif "abdominal" in question.lower() and ("x-ray" in question.lower() or "أشعة" in question):
            return "Abdominal X-ray" if "en" in question.lower() else "أشعة سينية على البطن"
        else:
            # If no specific type found, return a generic answer
            return "Radiograph" if "en" in question.lower() else "صورة إشعاعية"
    
    # Anatomy questions
    anatomy_keywords = ["where", "أين", "location", "موقع", "region", "منطقة"]
    if any(keyword in question.lower() for keyword in anatomy_keywords):
        # Apply medical translation to anatomical terms
        for eng, ar in MEDICAL_TRANSLATION_DICT.items():
            if eng in answer.lower():
                answer = answer.replace(eng, ar)
        return answer
    
    return answer

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
                    st.image(image, caption=T["upload_title"], use_container_width=True)
                    
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
                        "Is there evidence of pneumonia?",
                        "What kind of image is this?",
                        "Where is the abnormality located?"
                    ],
                    "ar": [
                        "ما هي التشوهات التي تراها؟",
                        "هل هناك أي كسور مرئية؟",
                        "هل هذه النتيجة طبيعية أم غير طبيعية؟",
                        "صف النتائج الطبية الرئيسية",
                        "هل هناك أي علامات للعدوى؟",
                        "هل هناك ورم أو كتلة مرئية؟",
                        "ما هو تقييمك التشخيصي؟",
                        "هل هناك دليل على الالتهاب الرئوي؟",
                        "ما نوع هذه الصورة؟",
                        "أين يقع الخلل؟"
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
                        
                        # Apply medical translation dictionary
                        arabic_answer = apply_medical_translation(arabic_answer)
                        
                        # Ensure answer is in Arabic
                        arabic_answer_display, arabic_translated = ensure_arabic_answer(arabic_answer)
                        
                        # Translate to English
                        with st.spinner("🌐 Translating results..." if st.session_state.lang == 'en' else "🌐 جاري ترجمة النتائج..."):
                            english_answer, _ = cached_translate_text(arabic_answer_display, "ar", "en")
                        
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
            st.subheader(T["features_title"])
            features_html = "<ul style='list-style-type: none; padding-left: 0;'>"
            for feature in T["features"]:
                # Extract icon and text
                icon = feature.split(" ")[0]
                text = " ".join(feature.split(" ")[1:])
                features_html += f"<li style='margin-bottom: 0.8rem;'>{icon} <strong>{text}</strong></li>"
            features_html += "</ul>"
            
            st.markdown(f"""
            <div style="background: linear-gradient(to bottom right, #e0f2fe, #dbeafe); 
                        padding: 1.2rem; border-radius: 0.8rem; margin-bottom: 1.5rem;">
                {features_html}
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.subheader(T["tech_title"])
            tech_html = "<ul style='list-style-type: none; padding-left: 0;'>"
            for tech in T["tech"]:
                # Extract icon and text
                icon = tech.split(" ")[0]
                text = " ".join(tech.split(" ")[1:])
                tech_html += f"<li style='margin-bottom: 0.8rem;'>{icon} <strong>{text}</strong></li>"
            tech_html += "</ul>"
            
            st.markdown(f"""
            <div style="background: linear-gradient(to bottom right, #ede9fe, #e0e7ff); 
                        padding: 1.2rem; border-radius: 0.8rem; margin-bottom: 1.5rem;">
                {tech_html}
            </div>
            """, unsafe_allow_html=True)
        
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
