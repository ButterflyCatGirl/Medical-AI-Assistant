# Ultra-Fast Medical VQA Streamlit App with Enhanced Translation Accuracy
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, pipeline
import logging
import time
import gc
from typing import Dict, Any
import warnings
import re
import requests
from bs4 import BeautifulSoup

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized Configuration
MAX_IMAGE_DIM = 512
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
BASE_MODEL = "ButterflyCatGirl/Blip-Streamlit-chatbot"

# Medical term dictionaries for validation
MEDICAL_TERMS_EN = {
    "fracture", "tumor", "infection", "pneumonia", "edema", 
    "cardiomegaly", "opacity", "consolidation", "effusion", 
    "nodule", "atelectasis", "cardiomegaly", "hernia"
}

MEDICAL_TERMS_AR = {
    "كسر", "ورم", "عدوى", "التهاب رئوي", "وذمة", 
    "تضخم القلب", "عتامة", "تصلب", "انصباب", 
    "عقدة", "انخماص", "فتق"
}

class AccurateMedicalVQA:
    """Medical VQA System with Enhanced Medical Translation"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = self._get_device()
        self.translation_models_loaded = False
        self.translation_cache = {}
        
    def _get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_translation_models(self):
        """Load specialized medical translation models"""
        try:
            logger.info("Loading medical translation models...")
            
            # Medical-optimized translation models
            self.en_ar_translator = pipeline(
                "translation", 
                model="Helsinki-NLP/opus-mt-en-medical",
                device=0 if self.device == "cuda" else -1
            )
            
            self.ar_en_translator = pipeline(
                "translation", 
                model="Helsinki-NLP/opus-mt-ar-en-medical",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Medical translation models loaded")
            self.translation_models_loaded = True
            return True
        except Exception as e:
            logger.error(f"Medical translation model loading failed: {str(e)}")
            return False
    
    def _validate_medical_translation(self, source: str, translation: str, source_lang: str) -> bool:
        """Validate medical terms in translation"""
        if source_lang == "en":
            source_terms = MEDICAL_TERMS_EN
            target_terms = MEDICAL_TERMS_AR
        else:
            source_terms = MEDICAL_TERMS_AR
            target_terms = MEDICAL_TERMS_EN
        
        # Check for critical medical terms
        for term in source_terms:
            if term in source.lower():
                if not any(t in translation.lower() for t in target_terms):
                    logger.warning(f"Medical term '{term}' not properly translated")
                    return False
        return True
    
    def _post_process_translation(self, text: str, target_lang: str) -> str:
        """Clean and format medical translations"""
        # Remove extra spaces and punctuation issues
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        text = re.sub(r'([.,;:])\s+', r'\1 ', text)
        
        # Medical-specific corrections
        corrections = {
            "x ray": "X-ray",
            "xray": "X-ray",
            "ct scan": "CT scan",
            "mri scan": "MRI scan",
            "اشعة اكس": "أشعة إكس",
            "اشعة مقطعية": "أشعة مقطعية"
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
            
        # Capitalize first letter for medical reports
        if text and target_lang == "en":
            text = text[0].upper() + text[1:]
            
        return text.strip()
    
    def translate_ar_to_en(self, text: str) -> str:
        """Translate Arabic to English with medical validation"""
        if not text.strip():
            return ""
        
        # Check cache first
        if text in self.translation_cache:
            return self.translation_cache[text]
        
        if not self.translation_models_loaded:
            self._load_translation_models()
            
        try:
            # Medical-optimized translation
            result = self.ar_en_translator(text)
            translation = result[0]['translation_text']
            
            # Validate medical terms
            if not self._validate_medical_translation(text, translation, "ar"):
                logger.warning(f"Retrying translation for: {text}")
                # Try a second time if validation fails
                result = self.ar_en_translator(text)
                translation = result[0]['translation_text']
            
            # Post-process
            translation = self._post_process_translation(translation, "en")
            
            # Cache result
            self.translation_cache[text] = translation
            return translation
        except Exception as e:
            logger.error(f"Arabic to English translation failed: {str(e)}")
            return text
    
    def translate_en_to_ar(self, text: str) -> str:
        """Translate English to Arabic with medical validation"""
        if not text.strip():
            return ""
        
        # Check cache first
        if text in self.translation_cache:
            return self.translation_cache[text]
        
        if not self.translation_models_loaded:
            self._load_translation_models()
        
        try:
            # Medical-optimized translation
            result = self.en_ar_translator(text)
            translation = result[0]['translation_text']
            
            # Validate medical terms
            if not self._validate_medical_translation(text, translation, "en"):
                logger.warning(f"Retrying translation for: {text}")
                # Try a second time if validation fails
                result = self.en_ar_translator(text)
                translation = result[0]['translation_text']
            
            # Post-process
            translation = self._post_process_translation(translation, "ar")
            
            # Cache result
            self.translation_cache[text] = translation
            return translation
        except Exception as e:
            logger.error(f"English to Arabic translation failed: {str(e)}")
            return text

    # ... rest of the class remains the same (load_model, _detect_language, etc) ...

# Streamlit Configuration
def init_app():
    st.set_page_config(
        page_title="Medical AI Assistant",
        layout="wide",
        page_icon="🩺",
        initial_sidebar_state="expanded"
    )

def apply_theme():
    st.markdown("""
    <style>
        /* ... existing styles ... */
        
        .translation-preview {
            background: #e8f4f8;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            border-left: 3px solid #2E8B57;
        }
        .term-highlight {
            background-color: #fff3cd;
            padding: 2px 4px;
            border-radius: 4px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# ... rest of the Streamlit app code remains the same ...

def main():
    """Main application"""
    init_app()
    apply_theme()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Accurate Medical AI Assistant</h1>
        <p><strong>Enhanced Translation Approach - Medical Image Analysis</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    vqa_system = get_vqa_system()
    
    # Load model
    if vqa_system.model is None:
        with st.spinner("🔄 Loading medical model..."):
            success = vqa_system.load_model()
            if success:
                st.success("✅ Medical model loaded successfully!")
                st.balloons()
            else:
                st.error("❌ Model loading failed. Please check the following:")
                st.error("1. Verify internet connection (models download from Hugging Face)")
                st.error("2. Try a smaller model or different approach")
                st.error("3. Check server logs for detailed error message")
                st.stop()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Medical Image")
        
        uploaded_file = st.file_uploader(
            "Choose medical image (max 5MB):",
            type=SUPPORTED_FORMATS,
            help="Supported: JPG, JPEG, PNG"
        )
        
        if uploaded_file:
            is_valid, message = validate_file(uploaded_file)
            
            if is_valid:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                    
                    # Show image stats
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.info(f"📊 Size: {image.size[0]}×{image.size[1]}")
                    with col_info2:
                        st.info(f"💾 Format: {uploaded_file.name.split('.')[-1].upper()}")
                except Exception as e:
                    st.error(f"❌ Image error: {str(e)}")
                    uploaded_file = None
            else:
                st.error(f"❌ {message}")
                uploaded_file = None
    
    with col2:
        st.markdown("### 💭 Ask Medical Question")
        
        # Language selector
        language = st.selectbox(
            "Language:",
            options=["ar", "en"],
            format_func=lambda x: "🇪🇬 العربية" if x == "ar" else "🇺🇸 English"
        )
        
        # Question input
        if language == "ar":
            placeholder = "ما التشخيص المحتمل لهذه الصورة؟ أو صف ما تراه في الصورة"
            label = "السؤال الطبي:"
        else:
            placeholder = "What is the likely diagnosis? Or describe what you see in the image"
            label = "Medical Question:"
        
        question = st.text_area(
            label,
            height=100,
            placeholder=placeholder
        )
        
        # Analyze button
        if st.button("🔍 Accurate Analysis"):
            if not uploaded_file:
                st.warning("⚠️ Upload image first")
            elif not question.strip():
                st.warning("⚠️ Enter question")
            else:
                with st.spinner("🧠 Analyzing with medical AI..."):
                    try:
                        image = Image.open(uploaded_file)
                        result = vqa_system.process_query(image, question)
                        
                        if result["success"]:
                            st.markdown("---")
                            st.markdown("### 🎯 Medical Analysis Results")
                            
                            # Processing time and accuracy indicator
                            st.markdown(f"""
                            <div class="accuracy-indicator">
                                ✅ <strong>Analysis Complete</strong> | 
                                ⏱️ <strong>{result['processing_time']:.2f}s</strong> | 
                                🔍 <strong>{'Arabic' if result['detected_language'] == 'ar' else 'English'}</strong> |
                                🎯 <strong>Confidence: {result['confidence']*100:.1f}%</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence visual
                            st.markdown(f"""
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {result['confidence']*100}%;"></div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Results
                            res_col1, res_col2 = st.columns(2)
                            
                            with res_col1:
                                st.markdown("**🇺🇸 English Analysis**")
                                st.markdown(f"**Q:** {result['question']}")
                                st.markdown(f"**Medical Finding:** {result['answer_en']}")
                            
                            with res_col2:
                                st.markdown("**🇪🇬 التحليل الطبي بالعربية**")
                                st.markdown(f"""
                                <div class="arabic-text">
                                    <strong>السؤال:</strong> {result['question']}<br><br>
                                    <strong>النتيجة الطبية:</strong> {result['answer_ar']}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Medical disclaimer
                            st.warning("⚠️ **للأغراض التعليمية فقط - استشر طبيب مختص للتشخيص النهائي**")
                            
                        else:
                            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown')}")
                    
                    except Exception as e:
                        st.error(f"❌ Processing error: {str(e)}")
    
    # Fixed indentation for sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        if vqa_system.model is not None:
            st.success("✅ Model: Ready")
            st.info(f"🖥️ Device: {vqa_system.device.upper()}")
            st.success("🌐 Translation Enabled")
        else:
            st.error("❌ Model: Not Ready")
        
        st.markdown("---")
        st.markdown("""
        **🔧 Translation Approach:**
        - Arabic questions → English → BLIP model
        - English answers → Arabic → Display
        
        **🎯 Accuracy Features:**
        - ✅ Medical-optimized prompts
        - ✅ Professional translation
        - ✅ Confidence scoring
        
        **📋 Best Practices:**
        1. Upload clear medical images
        2. Ask specific questions
        3. Use medical terminology
        4. Specify body parts/regions
        
        **🩺 Supported Analysis:**
        - X-rays, CT scans, MRI
        - Chest, brain, abdomen imaging
        - Bone fractures, infections
        - Tumors, fluid accumulation
        """)
        
        st.markdown("---")
        st.markdown("**⚠️ Medical Disclaimer**")
        st.caption("This AI provides preliminary analysis for educational purposes. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Medical VQA with Translation v2.0</strong> | Enhanced Arabic Support</p>
    </div>
    """, unsafe_allow_html=True)
