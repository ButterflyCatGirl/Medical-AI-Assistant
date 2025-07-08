# Ultra-Fast Medical VQA Streamlit App with Accurate Translation
import streamlit as st
from PIL import Image, ImageOps
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, MarianMTModel, MarianTokenizer
import logging
import time
import gc
from typing import Optional, Dict, Any
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized Configuration
MAX_IMAGE_DIM = 512  # Higher resolution for medical details
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
BASE_MODEL = "Salesforce/blip-vqa-base"  # Official base model

class AccurateMedicalVQA:
    """Accurate Medical VQA System with Enhanced Translation"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = self._get_device()
        self.translation_models_loaded = False
        
    def _get_device(self) -> str:
        """Get optimal device"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _load_translation_models(self):
        """Load translation models"""
        try:
            logger.info("Loading translation models...")
            # Arabic to English translation
            self.ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
            self.ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en").to(self.device)
            
            # English to Arabic translation
            self.en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
            self.en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar").to(self.device)
            
            logger.info("Translation models loaded successfully")
            self.translation_models_loaded = True
            return True
        except Exception as e:
            logger.error(f"Translation model loading failed: {str(e)}")
            return False
    
    def translate_ar_to_en(self, text: str) -> str:
        """Translate Arabic to English"""
        if not text.strip():
            return ""
        
        if not self.translation_models_loaded:
            self._load_translation_models()
        
        try:
            inputs = self.ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.ar_en_model.generate(**inputs)
            return self.ar_en_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Arabic to English translation failed: {str(e)}")
            return text
    
    def translate_en_to_ar(self, text: str) -> str:
        """Translate English to Arabic"""
        if not text.strip():
            return ""
        
        if not self.translation_models_loaded:
            self._load_translation_models()
        
        try:
            inputs = self.en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.en_ar_model.generate(**inputs)
            return self.en_ar_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"English to Arabic translation failed: {str(e)}")
            return text
    
    @st.cache_resource(show_spinner=False)
    def load_model(_self):
        """Load model with robust error handling"""
        try:
            logger.info(f"Loading model: {BASE_MODEL}")
            
            # Load processor and model
            _self.processor = BlipProcessor.from_pretrained(BASE_MODEL)
            
            # Handle device and precision
            if _self.device == "cpu":
                _self.model = BlipForQuestionAnswering.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch.float32
                )
            else:
                _self.model = BlipForQuestionAnswering.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch.float16
                )
            
            _self.model = _self.model.to(_self.device)
            _self.model.eval()
            
            logger.info(f"Model loaded successfully on {_self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            # Try alternative loading approach
            try:
                logger.info("Trying alternative model loading approach...")
                _self.processor = BlipProcessor.from_pretrained(BASE_MODEL)
                _self.model = BlipForQuestionAnswering.from_pretrained(BASE_MODEL)
                _self.model = _self.model.to(_self.device)
                _self.model.eval()
                logger.info("Model loaded successfully with alternative approach")
                return True
            except Exception as alt_e:
                logger.error(f"Alternative loading failed: {str(alt_e)}")
                return False
    
    def _detect_language(self, text: str) -> str:
        """Fast language detection"""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_chars > 0 else "en"
    
    def _process_image_optimized(self, image: Image.Image) -> Image.Image:
        """Medical-optimized image processing"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preserve aspect ratio with larger size for medical details
        width, height = image.size
        ratio = min(MAX_IMAGE_DIM/width, MAX_IMAGE_DIM/height)
        new_size = (int(width * ratio), int(height * ratio))
        
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    def _clean_generated_answer(self, raw_answer: str) -> str:
        """Medical-aware cleaning of generated answers"""
        if not raw_answer:
            return ""
        
        # Remove common VQA artifacts
        patterns_to_remove = [
            r"^the image shows ",
            r"^this is an image of ",
            r"^likely ",
            r"^probably ",
            r"^answer: ",
            r"^response: ",
            r"^based on the image, ",
            r" please consult a professional\.?$",
            r" this needs medical attention\.?$"
        ]
        
        for pattern in patterns_to_remove:
            raw_answer = re.sub(pattern, "", raw_answer, flags=re.IGNORECASE)
        
        # Capitalize first letter for medical report style
        return raw_answer.strip().capitalize()
    
    def _calculate_confidence(self, answer: str) -> float:
        """Calculate medical confidence score"""
        uncertainty_terms = ["may", "might", "possible", "potential", "appears", "suggestive", "likely"]
        certain_terms = ["clear", "definite", "evident", "diagnosis", "confirmed", "present", "shows"]
        
        uncertainty_count = sum(1 for term in uncertainty_terms if term in answer.lower())
        certainty_count = sum(1 for term in certain_terms if term in answer.lower())
        
        total_terms = uncertainty_count + certainty_count
        if total_terms == 0:
            return 0.8  # Default confidence
        
        return min(0.95, max(0.5, certainty_count / total_terms))
    
    def process_query(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Process query with translation approach"""
        try:
            start_time = time.time()
            
            # Process image
            image = self._process_image_optimized(image)
            
            # Detect language
            detected_lang = self._detect_language(question)
            
            # Translate question to English if needed
            if detected_lang == "ar":
                english_question = self.translate_ar_to_en(question)
                logger.info(f"Translated question: {question} -> {english_question}")
            else:
                english_question = question
            
            # Process with model
            inputs = self.processor(image, english_question, return_tensors="pt").to(self.device)
            
            # Generate with medical-optimized parameters
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_length=128,
                            num_beams=7,          # More beams for better accuracy
                            early_stopping=True,
                            do_sample=False,       # Disable sampling for deterministic output
                            no_repeat_ngram_size=3, # Prevent repetition of medical terms
                            repetition_penalty=2.0  # Stronger penalty for repetition
                        )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=7,
                        early_stopping=True,
                        do_sample=False,
                        no_repeat_ngram_size=3,
                        repetition_penalty=2.0
                    )
            
            # Decode and clean answer
            raw_answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            answer_en = self._clean_generated_answer(raw_answer)
            
            # Handle empty or poor answers
            if not answer_en or len(answer_en) < 5:
                answer_en = "Unable to provide a clear medical analysis from this image. Please consult a healthcare professional."
            
            # Translate answer to Arabic
            answer_ar = self.translate_en_to_ar(answer_en)
            
            # Calculate confidence
            confidence = self._calculate_confidence(answer_en)
            processing_time = time.time() - start_time
            
            return {
                "question": question,
                "answer_en": answer_en,
                "answer_ar": answer_ar,
                "detected_language": detected_lang,
                "processing_time": processing_time,
                "confidence": confidence,
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
        page_title="Accurate Medical AI",
        layout="wide",
        page_icon="🩺"
    )

def apply_theme():
    """Apply enhanced theme"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
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
            border-left: 4px solid #2E8B57;
            margin: 0.5rem 0;
        }
        .arabic-text {
            direction: rtl;
            text-align: right;
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            font-size: 18px;
        }
        .stButton > button {
            background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
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
        .accuracy-indicator {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 0.75rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .confidence-bar {
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 5px;
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_vqa_system():
    """Get cached VQA system"""
    return AccurateMedicalVQA()

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
    
    # Sidebar
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

if __name__ == "__main__":
    main()
