# Enhanced Medical VQA Streamlit App - Final Version with Proper Translation
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, MarianMTModel, MarianTokenizer
import logging
import time
import gc
from typing import Optional, Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_IMAGE_SIZE = (384, 384)
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
VQA_MODEL = "ButterflyCatGirl/Blip-Streamlit-chatbot"

class EnhancedMedicalVQA:
    """Enhanced Medical VQA System with Proper Translation"""
    
    def __init__(self):
        self.vqa_processor = None
        self.vqa_model = None
        self.ar_en_tokenizer = None
        self.ar_en_model = None
        self.en_ar_tokenizer = None
        self.en_ar_model = None
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Get optimal device"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @st.cache_resource(show_spinner=False)
    def load_models(_self):
        """Load all models with caching"""
        try:
            logger.info("Loading VQA model...")
            # Load VQA model
            _self.vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL)
            _self.vqa_processor = BlipProcessor.from_pretrained(VQA_MODEL)
            _self.vqa_model = _self.vqa_model.to(_self.device)
            _self.vqa_model.eval()
            
            logger.info("Loading translation models...")
            # Load Arabic to English translation
            _self.ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
            _self.ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
            _self.ar_en_model = _self.ar_en_model.to(_self.device)
            
            # Load English to Arabic translation
            _self.en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
            _self.en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
            _self.en_ar_model = _self.en_ar_model.to(_self.device)
            
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False
    
    def _detect_language(self, text: str) -> str:
        """Detect language"""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_chars > 0 else "en"
    
    def translate_ar_to_en(self, text: str) -> str:
        """Translate Arabic to English"""
        try:
            inputs = self.ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated = self.ar_en_model.generate(**inputs, max_length=512)
            
            return self.ar_en_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Arabic to English translation failed: {str(e)}")
            return text
    
    def translate_en_to_ar(self, text: str) -> str:
        """Translate English to Arabic"""
        try:
            inputs = self.en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated = self.en_ar_model.generate(**inputs, max_length=512)
            
            return self.en_ar_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"English to Arabic translation failed: {str(e)}")
            return text
    
    def _process_image(self, image: Image.Image) -> Image.Image:
        """Process image for optimal performance"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if image.size != MAX_IMAGE_SIZE:
            image = image.resize(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        return image
    
    def process_medical_query(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Process medical VQA query with proper translation"""
        try:
            start_time = time.time()
            
            if not question.strip():
                return {"error": "السؤال فارغ", "success": False}
            
            # Process image
            image = self._process_image(image)
            
            # Detect language and translate if needed
            detected_lang = self._detect_language(question)
            
            if detected_lang == "ar":
                # Translate Arabic question to English
                english_question = self.translate_ar_to_en(question)
                original_question = question
            else:
                english_question = question
                original_question = question
            
            # Process with VQA model
            inputs = self.vqa_processor(image, english_question, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        output = self.vqa_model.generate(
                            **inputs,
                            max_length=150,
                            num_beams=4,
                            early_stopping=True,
                            temperature=0.7,
                            do_sample=True
                        )
                else:
                    output = self.vqa_model.generate(
                        **inputs,
                        max_length=150,
                        num_beams=4,
                        early_stopping=True,
                        temperature=0.7,
                        do_sample=True
                    )
            
            # Decode English answer
            english_answer = self.vqa_processor.decode(output[0], skip_special_tokens=True)
            
            # Clean answer
            english_answer = english_answer.strip()
            if not english_answer or len(english_answer) < 3:
                english_answer = "No clear medical findings can be determined from this image."
            
            # Translate answer if original question was in Arabic
            if detected_lang == "ar":
                arabic_answer = self.translate_en_to_ar(english_answer)
            else:
                arabic_answer = self.translate_en_to_ar(english_answer)
            
            processing_time = time.time() - start_time
            
            return {
                "original_question": original_question,
                "english_question": english_question if detected_lang == "ar" else question,
                "english_answer": english_answer,
                "arabic_answer": arabic_answer,
                "detected_language": detected_lang,
                "processing_time": processing_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

# Streamlit Configuration
def init_app():
    """Initialize app"""
    st.set_page_config(
        page_title="Enhanced Medical VQA",
        layout="wide",
        page_icon="🩺"
    )

def apply_theme():
    """Apply theme"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .result-container {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #2E86AB;
            margin: 1rem 0;
        }
        .arabic-text {
            direction: rtl;
            text-align: right;
            font-family: 'Arial', sans-serif;
            font-size: 1.1em;
            line-height: 1.6;
        }
        .english-text {
            font-size: 1.1em;
            line-height: 1.6;
        }
        .stButton > button {
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.8rem 2rem;
            font-size: 1.1em;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(46, 134, 171, 0.4);
        }
        .stats-container {
            background: #e8f5e8;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_vqa_system():
    """Get cached VQA system"""
    return EnhancedMedicalVQA()

def validate_file(uploaded_file) -> tuple:
    """Validate uploaded file"""
    if not uploaded_file:
        return False, "No file uploaded"
    
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, "File too large (max 5MB)"
    
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        return False, f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
    
    return True, "Valid file"

def main():
    """Main application"""
    init_app()
    apply_theme()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Enhanced Medical VQA Assistant</h1>
        <p><strong>Advanced AI-Powered Medical Image Analysis with Bilingual Support</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    vqa_system = get_vqa_system()
    
    # Load models
    if vqa_system.vqa_model is None:
        with st.spinner("🚀 Loading advanced models (VQA + Translation)..."):
            success = vqa_system.load_models()
            if success:
                st.success("✅ All models loaded successfully! Ready for analysis!")
                st.balloons()
            else:
                st.error("❌ Failed to load models")
                st.stop()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Medical Image")
        
        uploaded_file = st.file_uploader(
            "Choose medical image:",
            type=SUPPORTED_FORMATS,
            help="Supported formats: JPG, JPEG, PNG (max 5MB)"
        )
        
        if uploaded_file:
            is_valid, message = validate_file(uploaded_file)
            
            if is_valid:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                    st.info(f"📊 Resolution: {image.size[0]}×{image.size[1]} pixels")
                except Exception as e:
                    st.error(f"❌ Image error: {str(e)}")
                    uploaded_file = None
            else:
                st.error(f"❌ {message}")
                uploaded_file = None
    
    with col2:
        st.markdown("### 💭 Ask Your Medical Question")
        
        # Language selector
        language = st.selectbox(
            "Select Language:",
            options=["ar", "en"],
            format_func=lambda x: "🇸🇦 العربية (Arabic)" if x == "ar" else "🇺🇸 English"
        )
        
        # Question input
        if language == "ar":
            placeholder = "مثال: ما هو التشخيص المحتمل لهذه الصورة؟ هل تظهر أي علامات مرضية؟"
            label = "السؤال الطبي:"
        else:
            placeholder = "Example: What is the likely diagnosis? Are there any abnormal findings?"
            label = "Medical Question:"
        
        question = st.text_area(
            label,
            height=120,
            placeholder=placeholder,
            help="Ask specific medical questions about the uploaded image"
        )
        
        # Analyze button
        if st.button("🔍 Analyze Medical Image"):
            if not uploaded_file:
                st.warning("⚠️ Please upload a medical image first")
            elif not question.strip():
                st.warning("⚠️ Please enter your medical question")
            else:
                with st.spinner("🔬 Analyzing medical image..."):
                    try:
                        image = Image.open(uploaded_file)
                        result = vqa_system.process_medical_query(image, question)
                        
                        if result["success"]:
                            st.markdown("---")
                            st.markdown("### 📋 Medical Analysis Results")
                            
                            # Processing stats
                            st.markdown(f"""
                            <div class="stats-container">
                                ⏱️ <strong>Processing Time:</strong> {result['processing_time']:.2f} seconds | 
                                🌐 <strong>Detected Language:</strong> {'Arabic' if result['detected_language'] == 'ar' else 'English'}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Results
                            res_col1, res_col2 = st.columns(2)
                            
                            with res_col1:
                                st.markdown("#### 🇺🇸 English Analysis")
                                st.markdown(f"""
                                <div class="result-container">
                                    <div class="english-text">
                                        <strong>Question:</strong> {result['english_question']}<br><br>
                                        <strong>Medical Analysis:</strong> {result['english_answer']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with res_col2:
                                st.markdown("#### 🇸🇦 التحليل باللغة العربية")
                                st.markdown(f"""
                                <div class="result-container">
                                    <div class="arabic-text">
                                        <strong>السؤال:</strong> {result['original_question']}<br><br>
                                        <strong>التحليل الطبي:</strong> {result['arabic_answer']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Medical disclaimer
                            st.error("⚠️ **تنبيه طبي مهم: هذا التحليل للأغراض التعليمية فقط - يجب استشارة طبيب مختص للتشخيص النهائي**")
                            st.error("⚠️ **Medical Disclaimer: This analysis is for educational purposes only - consult a qualified physician for final diagnosis**")
                            
                        else:
                            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Processing error: {str(e)}")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        if vqa_system.vqa_model is not None:
            st.success("✅ VQA Model: Ready")
            st.success("✅ Translation Models: Ready")
            st.info(f"🖥️ Device: {vqa_system.device.upper()}")
        else:
            st.error("❌ Models: Not Ready")
        
        st.markdown("---")
        st.markdown("""
        ### 🚀 Enhanced Features:
        - ✅ Specialized medical VQA model
        - ✅ Professional translation (MarianMT)
        - ✅ Bilingual support (Arabic/English)
        - ✅ Advanced image processing
        - ✅ Optimized inference
        
        ### 📖 How to Use:
        1. Upload medical image (X-ray, CT, MRI, etc.)
        2. Choose your preferred language
        3. Ask specific medical questions
        4. Get detailed bilingual analysis
        
        ### 📋 Specifications:
        - **Languages:** Arabic, English
        - **Image Formats:** JPG, PNG, JPEG
        - **Max File Size:** 5MB
        - **Resolution:** Auto-optimized
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Important Medical Notice")
        st.caption("""
        This AI system is designed for educational and research purposes only. 
        It should NOT replace professional medical consultation, diagnosis, or treatment. 
        Always consult qualified healthcare professionals for medical decisions.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>Enhanced Medical VQA System v2.0</strong></p>
        <p>Powered by Advanced AI Models | Professional Translation | Bilingual Support</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
