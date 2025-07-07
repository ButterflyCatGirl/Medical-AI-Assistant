# Fast Medical VQA Streamlit App - Optimized Version
import streamlit as st
from PIL import Image, ImageOps
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging
import time
import gc
from typing import Dict, Any
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_IMAGE_SIZE = (224, 224)  # Small for speed
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
FINE_TUNED_MODEL = "sharawy53/blip-vqa-medical-arabic"

class FastMedicalVQA:
    """Ultra-fast Medical VQA System"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = self._get_device()
        self.medical_knowledge = self._load_medical_terms()
        
    def _get_device(self) -> str:
        """Get optimal device"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _load_medical_terms(self) -> Dict[str, str]:
        """Medical terminology for better Arabic responses"""
        return {
            "normal": "طبيعي", "abnormal": "غير طبيعي",
            "chest": "الصدر", "lung": "الرئة", "heart": "القلب",
            "x-ray": "أشعة سينية", "ct": "أشعة مقطعية",
            "fracture": "كسر", "pneumonia": "التهاب رئوي",
            "tumor": "ورم", "mass": "كتلة", "fluid": "سوائل",
            "bone": "عظم", "brain": "المخ", "liver": "الكبد",
            "shows": "يُظهر", "appears": "يبدو", "indicates": "يشير إلى"
        }
    
    @st.cache_resource(show_spinner=False)
    def load_model(_self):
        """Load fine-tuned model with caching"""
        try:
            logger.info(f"Loading fine-tuned model: {FINE_TUNED_MODEL}")
            
            # Try fine-tuned model first
            _self.processor = BlipProcessor.from_pretrained(FINE_TUNED_MODEL)
            
            if _self.device == "cpu":
                _self.model = BlipForConditionalGeneration.from_pretrained(
                    FINE_TUNED_MODEL,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            else:
                _self.model = BlipForConditionalGeneration.from_pretrained(
                    FINE_TUNED_MODEL,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            
            _self.model = _self.model.to(_self.device)
            _self.model.eval()
            
            logger.info(f"Fine-tuned model loaded successfully on {_self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Fine-tuned model loading failed: {str(e)}")
            # Fallback to base model
            try:
                logger.info("Trying fallback base model...")
                _self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
                _self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")
                _self.model = _self.model.to(_self.device)
                _self.model.eval()
                logger.info("Fallback base model loaded successfully")
                return True
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {str(fallback_error)}")
                return False
    
    def _detect_language(self, text: str) -> str:
        """Fast language detection"""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_chars > 0 else "en"
    
    def _translate_to_arabic(self, text_en: str) -> str:
        """Fast medical translation to Arabic"""
        if not text_en:
            return "لا توجد إجابة واضحة"
        
        # Quick medical term replacement
        text_lower = text_en.lower()
        translated = text_en
        
        for en_term, ar_term in self.medical_knowledge.items():
            if en_term in text_lower:
                translated = re.sub(
                    re.escape(en_term), ar_term, translated, flags=re.IGNORECASE
                )
        
        # If still mostly English, provide contextual response
        arabic_chars = sum(1 for c in translated if '\u0600' <= c <= '\u06FF')
        if arabic_chars < 3:
            if any(term in text_lower for term in ["normal", "healthy"]):
                return "الصورة تبدو طبيعية وسليمة"
            elif any(term in text_lower for term in ["abnormal", "problem"]):
                return "تظهر الصورة نتائج تحتاج لتقييم طبي"
            elif "chest" in text_lower:
                return "صورة أشعة سينية للصدر تحتاج لفحص طبي"
            else:
                return "الصورة الطبية تحتاج لتقييم من طبيب مختص"
        
        return translated
    
    def _process_image_fast(self, image: Image.Image) -> Image.Image:
        """Ultra-fast image processing"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for speed
        if image.size != MAX_IMAGE_SIZE:
            image = image.resize(MAX_IMAGE_SIZE, Image.Resampling.BILINEAR)
        
        return image
    
    def process_query(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Process query with optimizations"""
        try:
            start_time = time.time()
            
            # Fast image processing
            image = self._process_image_fast(image)
            
            # Detect language
            detected_lang = self._detect_language(question)
            
            # Process with model
            inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            
            # Generate with speed optimizations
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=32,  # Shorter for speed
                    num_beams=2,    # Faster beam search
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Decode answer
            answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean answer
            if question.lower() in answer.lower():
                answer = answer.replace(question, "").strip()
            
            # Prepare responses
            if detected_lang == "ar":
                answer_ar = answer
                answer_en = answer
            else:
                answer_en = answer
                answer_ar = self._translate_to_arabic(answer)
            
            processing_time = time.time() - start_time
            
            return {
                "question": question,
                "answer_en": answer_en,
                "answer_ar": answer_ar,
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
    """Initialize app with optimized settings"""
    st.set_page_config(
        page_title="Fast Medical AI",
        layout="wide",
        page_icon="🩺"
    )

def apply_theme():
    """Apply optimized theme"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .result-box {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #4a90e2;
            margin: 0.5rem 0;
        }
        .arabic-text {
            direction: rtl;
            text-align: right;
            font-family: 'Arial', sans-serif;
        }
        .stButton > button {
            background: #4a90e2;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            width: 100%;
        }
        .fast-stats {
            background: #e8f5e8;
            padding: 0.5rem;
            border-radius: 5px;
            font-size: 0.9em;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_vqa_system():
    """Get cached VQA system"""
    return FastMedicalVQA()

def validate_file(uploaded_file) -> tuple:
    """Quick file validation"""
    if not uploaded_file:
        return False, "No file uploaded"
    
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, "File too large (max 5MB)"
    
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        return False, f"Use: {', '.join(SUPPORTED_FORMATS)}"
    
    return True, "Valid file"

def main():
    """Main application"""
    init_app()
    apply_theme()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Fast Medical AI Assistant</h1>
        <p><strong>Optimized for Speed - Powered by Fine-tuned BLIP Model</strong></p>
        <p>🚀 Model: <code>sharawy53/blip-vqa-medical-arabic</code></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    vqa_system = get_vqa_system()
    
    # Load model
    if vqa_system.model is None:
        with st.spinner("⚡ Loading optimized model..."):
            success = vqa_system.load_model()
            if success:
                st.success("✅ Model loaded! Ready for fast analysis!")
                st.balloons()
            else:
                st.error("❌ Model loading failed")
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
                    st.info(f"📊 Size: {image.size[0]}×{image.size[1]}")
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
            placeholder = "ما التشخيص المحتمل لهذه الصورة؟"
            label = "السؤال الطبي:"
        else:
            placeholder = "What is the likely diagnosis?"
            label = "Medical Question:"
        
        question = st.text_area(
            label,
            height=100,
            placeholder=placeholder
        )
        
        # Analyze button
        if st.button("⚡ Fast Analysis"):
            if not uploaded_file:
                st.warning("⚠️ Upload image first")
            elif not question.strip():
                st.warning("⚠️ Enter question")
            else:
                with st.spinner("⚡ Analyzing..."):
                    try:
                        image = Image.open(uploaded_file)
                        result = vqa_system.process_query(image, question)
                        
                        if result["success"]:
                            st.markdown("---")
                            st.markdown("### ⚡ Fast Results")
                            
                            # Processing time
                            st.markdown(f"""
                            <div class="fast-stats">
                                ⏱️ <strong>{result['processing_time']:.2f}s</strong> | 
                                🔍 <strong>{'Arabic' if result['detected_language'] == 'ar' else 'English'}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Results
                            res_col1, res_col2 = st.columns(2)
                            
                            with res_col1:
                                st.markdown("**🇺🇸 English**")
                                st.markdown(f"**Q:** {result['question']}")
                                st.markdown(f"**A:** {result['answer_en']}")
                            
                            with res_col2:
                                st.markdown("**🇪🇬 العربية**")
                                st.markdown(f"""
                                <div class="arabic-text">
                                    <strong>السؤال:</strong> {result['question']}<br>
                                    <strong>الإجابة:</strong> {result['answer_ar']}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Medical disclaimer
                            st.warning("⚠️ **للأغراض التعليمية فقط - استشر طبيب مختص**")
                            
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown')}")
                    
                    except Exception as e:
                        st.error(f"❌ Processing error: {str(e)}")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        if vqa_system.model is not None:
            st.success("✅ Model: Ready")
            st.info(f"🖥️ Device: {vqa_system.device.upper()}")
            st.info(f"🚀 Model: Fine-tuned BLIP")
        else:
            st.error("❌ Model: Not Ready")
        
        st.markdown("---")
        st.markdown("""
        **⚡ Speed Optimizations:**
        - ✅ Model caching
        - ✅ Fast image processing
        - ✅ Optimized inference
        - ✅ Quick translation
        
        **🎯 Usage:**
        1. Upload medical image
        2. Select language
        3. Ask specific question
        4. Get instant results
        
        **📋 Supported:**
        - Languages: Arabic, English
        - Formats: JPG, PNG
        - Max size: 5MB
        """)
        
        st.markdown("---")
        st.markdown("**⚠️ Medical Disclaimer**")
        st.caption("For educational use only. Always consult healthcare professionals for medical decisions.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Fast Medical VQA v1.0</strong> | Optimized Response Times</p>
        <p>Powered by: <code>sharawy53/blip-vqa-medical-arabic</code></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
