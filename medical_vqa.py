# Ultra-Fast Medical VQA Streamlit App - FINAL ACCURATE VERSION
import streamlit as st
from PIL import Image, ImageOps
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
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
FINE_TUNED_MODEL = "ButterflyCatGirl/Blip-Streamlit-chatbot"
MEDICAL_FALLBACK_MODEL = "blip-vqa-medical-final"

class AccurateMedicalVQA:
    """Accurate Medical VQA System with Enhanced Responses"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.translator = None
        self.device = self._get_device()
        self.medical_terms = self._load_comprehensive_medical_terms()
        self.translator_loaded = False
        
    def _get_device(self) -> str:
        """Get optimal device"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _load_comprehensive_medical_terms(self) -> Dict[str, str]:
        """Comprehensive medical terminology for accurate Arabic responses"""
        return {
            # Basic medical terms
            "normal": "طبيعي", "abnormal": "غير طبيعي", "healthy": "سليم",
            "disease": "مرض", "condition": "حالة", "patient": "مريض",
            
            # Body parts and anatomy
            "chest": "الصدر", "lung": "الرئة", "lungs": "الرئتان", "heart": "القلب",
            "brain": "الدماغ", "liver": "الكبد", "kidney": "الكلية", "spine": "العمود الفقري",
            "bone": "عظم", "bones": "عظام", "skull": "الجمجمة", "rib": "ضلع", "ribs": "أضلاع",
            "abdomen": "البطن", "pelvis": "الحوض", "shoulder": "الكتف", "neck": "الرقبة",
            
            # Medical imaging
            "x-ray": "أشعة سينية", "ct scan": "تصوير مقطعي محوسب", "mri": "رنين مغناطيسي",
            "ultrasound": "موجات فوق صوتية", "radiograph": "صورة شعاعية", "scan": "فحص بالأشعة",
            
            # Medical conditions
            "pneumonia": "التهاب رئوي", "infection": "التهاب", "inflammation": "التهاب",
            "fracture": "كسر", "broken": "مكسور", "tumor": "ورم", "mass": "كتلة",
            "cancer": "سرطان", "fluid": "سوائل", "swelling": "تورم", "pain": "ألم",
            "hemorrhage": "نزيف", "edema": "وذمة", "atelectasis": "انخماص",
            
            # Medical observations
            "shows": "يُظهر", "appears": "يبدو", "indicates": "يشير إلى", "suggests": "يوحي بـ",
            "visible": "مرئي", "evident": "واضح", "present": "موجود", "absent": "غائب",
            "enlarged": "متضخم", "reduced": "منخفض", "increased": "مرتفع", "decreased": "منخفض",
            "opacity": "عتامة", "consolidation": "تجمع",
            
            # Medical actions
            "examination": "فحص", "diagnosis": "تشخيص", "treatment": "علاج", "surgery": "جراحة",
            "consultation": "استشارة", "follow-up": "متابعة", "monitoring": "مراقبة",
            
            # Common phrases
            "what is": "ما هو", "what are": "ما هي", "this image": "هذه الصورة",
            "medical image": "صورة طبية", "likely": "محتمل", "possible": "ممكن",
            "finding": "نتيجة", "observation": "ملاحظة"
        }
    
    def _create_medical_prompt(self, question: str) -> str:
        """Create concise medical prompt for better responses"""
        return f"MEDICAL ANALYSIS: {question} [Provide precise medical observations only]"
    
    @st.cache_resource(show_spinner=False)
    def load_model(_self):
        """Load fine-tuned model with caching"""
        try:
            logger.info(f"Loading fine-tuned model: {FINE_TUNED_MODEL}")
            
            # Load with optimizations
            _self.processor = BlipProcessor.from_pretrained(FINE_TUNED_MODEL)
            
            # Set pad token properly
            if _self.processor.tokenizer.pad_token is None:
                _self.processor.tokenizer.pad_token = _self.processor.tokenizer.eos_token
            
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
            
            logger.info(f"Model loaded successfully on {_self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Primary model loading failed: {str(e)}")
            # Fallback to medical-specific model
            try:
                _self.processor = BlipProcessor.from_pretrained(MEDICAL_FALLBACK_MODEL)
                
                if _self.device == "cpu":
                    _self.model = BlipForConditionalGeneration.from_pretrained(
                        MEDICAL_FALLBACK_MODEL,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                else:
                    _self.model = BlipForConditionalGeneration.from_pretrained(
                        MEDICAL_FALLBACK_MODEL,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
                
                _self.model = _self.model.to(_self.device)
                _self.model.eval()
                logger.info(f"Fallback to medical model {MEDICAL_FALLBACK_MODEL} successful")
                return True
            except Exception as fallback_error:
                logger.error(f"Medical fallback model failed: {str(fallback_error)}")
                return False
    
    def _detect_language(self, text: str) -> str:
        """Fast language detection"""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_chars > 0 else "en"
    
    def _load_translator(self):
        """Lazy load translation model"""
        if not self.translator_loaded:
            try:
                device_id = 0 if self.device == "cuda" else -1
                self.translator = pipeline(
                    "translation_en_to_ar",
                    model="Helsinki-NLP/opus-mt-en-ar",
                    device=device_id,
                    max_length=256
                )
                self.translator_loaded = True
                logger.info("Medical translator loaded successfully")
            except Exception as e:
                logger.error(f"Translator loading failed: {str(e)}")
                self.translator = None
                self.translator_loaded = True
    
    def _rule_based_arabic_translation(self, text_en: str) -> str:
        """Rule-based fallback translation"""
        words = text_en.split()
        translated_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word.lower())
            if clean_word in self.medical_terms:
                translated_words.append(self.medical_terms[clean_word])
            else:
                # Check for partial matches
                found = False
                for en_term, ar_term in self.medical_terms.items():
                    if en_term in clean_word or clean_word in en_term:
                        translated_words.append(ar_term)
                        found = True
                        break
                if not found:
                    translated_words.append(word)
        
        result = " ".join(translated_words)
        
        # If translation is still poor, provide contextual medical response
        arabic_char_count = sum(1 for c in result if '\u0600' <= c <= '\u06FF')
        if arabic_char_count < 3:
            return "تحتاج هذه الصورة الطبية إلى تحليل وتفسير من قبل طبيب مختص في الأشعة"
        
        return result
    
    def _translate_to_arabic_medical(self, text_en: str) -> str:
        """Advanced medical translation to Arabic"""
        if not text_en or text_en.strip() == "":
            return "لا يمكن تحديد النتائج بوضوح من الصورة"
        
        # Clean the text first
        text_clean = text_en.strip()
        
        # Try to load translator if not already loaded
        if not self.translator_loaded:
            self._load_translator()
        
        # Use proper translator if available
        if self.translator is not None:
            try:
                return self.translator(text_clean, max_length=256)[0]['translation_text']
            except Exception as e:
                logger.error(f"Translation failed: {str(e)}, using rule-based fallback")
                return self._rule_based_arabic_translation(text_clean)
        else:
            return self._rule_based_arabic_translation(text_clean)
    
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
        """Process query with enhanced accuracy"""
        try:
            start_time = time.time()
            
            # Process image with medical-optimized processing
            image = self._process_image_optimized(image)
            
            # Detect language
            detected_lang = self._detect_language(question)
            
            # Create enhanced medical prompt
            enhanced_question = self._create_medical_prompt(question)
            
            # Process with model
            inputs = self.processor(image, enhanced_question, return_tensors="pt").to(self.device)
            
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
            
            # Generate Arabic response
            if detected_lang == "ar":
                answer_ar = self._translate_to_arabic_medical(answer_en)
            else:
                answer_ar = self._translate_to_arabic_medical(answer_en)
            
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
        <p><strong>Enhanced for Precision - Advanced Medical Image Analysis</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    vqa_system = get_vqa_system()
    
    # Load model
    if vqa_system.model is None:
        with st.spinner("🔄 Loading enhanced medical model..."):
            success = vqa_system.load_model()
            if success:
                st.success("✅ Advanced medical model loaded successfully!")
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
                with st.spinner("🧠 Analyzing with enhanced AI..."):
                    try:
                        image = Image.open(uploaded_file)
                        result = vqa_system.process_query(image, question)
                        
                        if result["success"]:
                            st.markdown("---")
                            st.markdown("### 🎯 Accurate Medical Analysis")
                            
                            # Processing time and accuracy indicator
                            st.markdown(f"""
                            <div class="accuracy-indicator">
                                ✅ <strong>Enhanced Analysis Complete</strong> | 
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
            st.success("🎯 Enhanced Accuracy Mode")
        else:
            st.error("❌ Model: Not Ready")
        
        st.markdown("---")
        st.markdown("""
        **🎯 Accuracy Features:**
        - ✅ Medical-optimized prompts
        - ✅ Deterministic response generation
        - ✅ Professional Arabic translation
        - ✅ Medical context awareness
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
        - Normal vs abnormal findings
        - Tumors, fluid accumulation
        """)
        
        st.markdown("---")
        st.markdown("**⚠️ Medical Disclaimer**")
        st.caption("This AI provides preliminary analysis for educational purposes. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Accurate Medical VQA v2.0</strong> | Enhanced Precision & Arabic Support</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
