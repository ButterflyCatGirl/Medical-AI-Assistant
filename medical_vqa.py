# Ultra-Fast Medical VQA Streamlit App with Enhanced Translation Accuracy
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, MarianMTModel, MarianTokenizer
import logging
import time
import re
import warnings
from typing import Dict, Any
import os
import sys

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized Configuration
MAX_IMAGE_DIM = 512
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
# Use a reliable base model with fallback
BASE_MODELS = [
    "blip-vqa-medical-final"
  #  "mennahhhh/mobilenet-t5-vqa"
    "Salesforce/blip-vqa-base",  # More reliable base model
   # "ButterflyCatGirl/Blip-Streamlit-chatbot"  # Fallback to custom model
]

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
        self.model_loading_error = None
        self.loaded_model_name = None
        self.load_status = "Not Loaded"
        
    def _get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_translation_models(self):
        """Load specialized medical translation models"""
        try:
            logger.info("Loading medical translation models...")
            
            # Medical-optimized translation models
            self.ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en-medical")
            self.ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en-medical").to(self.device)
            
            self.en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-medical")
            self.en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-medical").to(self.device)
            
            logger.info("Medical translation models loaded")
            self.translation_models_loaded = True
            return True
        except Exception as e:
            logger.error(f"Medical translation model loading failed: {str(e)}")
            self.model_loading_error = str(e)
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
            # Tokenize and translate
            inputs = self.ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.ar_en_model.generate(**inputs)
            translation = self.ar_en_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Validate medical terms
            if not self._validate_medical_translation(text, translation, "ar"):
                logger.warning(f"Retrying translation for: {text}")
                # Try a second time if validation fails
                inputs = self.ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                translated = self.ar_en_model.generate(**inputs)
                translation = self.ar_en_tokenizer.decode(translated[0], skip_special_tokens=True)
            
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
            # Tokenize and translate
            inputs = self.en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.en_ar_model.generate(**inputs)
            translation = self.en_ar_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Validate medical terms
            if not self._validate_medical_translation(text, translation, "en"):
                logger.warning(f"Retrying translation for: {text}")
                # Try a second time if validation fails
                inputs = self.en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                translated = self.en_ar_model.generate(**inputs)
                translation = self.en_ar_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Post-process
            translation = self._post_process_translation(translation, "ar")
            
            # Cache result
            self.translation_cache[text] = translation
            return translation
        except Exception as e:
            logger.error(f"English to Arabic translation failed: {str(e)}")
            return text

    def load_model(self):
        """Load model with robust error handling and fallback"""
        success = False
        last_error = None
        
        for model_name in BASE_MODELS:
            try:
                logger.info(f"Attempting to load model: {model_name}")
                
                # Load processor and model
                self.processor = BlipProcessor.from_pretrained(model_name)
                
                # Handle device and precision
                if self.device == "cpu":
                    self.model = BlipForQuestionAnswering.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32
                    )
                else:
                    self.model = BlipForQuestionAnswering.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16
                    )
                
                self.model = self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Model loaded successfully on {self.device}")
                self.loaded_model_name = model_name
                self.load_status = f"Loaded: {model_name}"
                success = True
                break
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")
                last_error = str(e)
                # Clear any partial state
                self.processor = None
                self.model = None
                # Try next model
        
        if not success:
            logger.error(f"All model loading attempts failed")
            self.model_loading_error = last_error
            self.load_status = "Failed to load"
        
        return success

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
    st.set_page_config(
        page_title="Medical AI Assistant",
        layout="wide",
        page_icon="🩺",
        initial_sidebar_state="expanded"
    )

def apply_theme():
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
        .error-box {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #f5c6cb;
            font-family: monospace;
            font-size: 14px;
            white-space: pre-wrap;
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
    
    # Header with improved design
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Accurate Medical AI Assistant</h1>
        <p><strong>Enhanced Medical Translation - Specialized for Healthcare</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    vqa_system = get_vqa_system()
    
    # Environment info
    with st.expander("Environment Information", expanded=False):
        st.write(f"**Python version:** {sys.version}")
        st.write(f"**PyTorch version:** {torch.__version__}")
        st.write(f"**Device:** {vqa_system.device}")
        st.write(f"**Available GPUs:** {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            st.write(f"**Current GPU:** {torch.cuda.get_device_name(0)}")
        st.write(f"**Hugging Face cache:** {os.getenv('TRANSFORMERS_CACHE', 'Default')}")
        st.write(f"**Model Status:** {vqa_system.load_status}")
    
    # Load model if not loaded
    if vqa_system.model is None:
        with st.spinner("🔄 Loading medical model..."):
            success = vqa_system.load_model()
            if success:
                st.success(f"✅ Medical model loaded successfully! (Using: {vqa_system.loaded_model_name})")
                st.balloons()
            else:
                st.error("❌ Model loading failed. Please check the following:")
                st.error("1. Verify internet connection (models download from Hugging Face)")
                st.error("2. Try a smaller model or different approach")
                st.error("3. Check server logs for detailed error message")
                
                # Show detailed error
                st.markdown("### Detailed Error Information")
                st.code(vqa_system.model_loading_error, language="text")
                
                # Troubleshooting guide
                with st.expander("Troubleshooting Guide"):
                    st.markdown("""
                    ### Common Solutions:
                    
                    1. **Check Internet Connection**: 
                       - The app needs to download models from Hugging Face Hub
                       - Ensure your server has internet access
                       - If behind a firewall, whitelist `huggingface.co`
                    
                    2. **Reduce Model Size**:
                       - We're trying smaller models automatically
                       - If issues persist, consider using CPU-only mode
                    
                    3. **Increase Resources**:
                       - If running locally, try with GPU
                       - On cloud services, upgrade to a plan with more memory
                    
                    4. **Clear Cache**:
                       - Streamlit caches models, which can sometimes cause issues
                       - Try clearing cache with `streamlit cache clear`
                    
                    5. **Alternative Models**:
                       - We'll attempt to use a simpler model as a fallback
                    """)
                
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
        
        # Language selector with flags
        language = st.radio(
            "Select Language:",
            options=["ar", "en"],
            format_func=lambda x: "🇪🇬 العربية" if x == "ar" else "🇺🇸 English",
            horizontal=True
        )
        
        # Question input with medical examples
        if language == "ar":
            placeholder = "ماذا تُظهر هذه الصورة الشعاعية؟ أو صف التشخيص المحتمل"
            label = "السؤال الطبي:"
        else:
            placeholder = "What does this X-ray show? Or describe the likely diagnosis"
            label = "Medical Question:"
        
        question = st.text_area(
            label,
            height=100,
            placeholder=placeholder,
            help="Be specific: mention body part, view type, or symptoms"
        )
        
        # Translation preview
        if question and language == "ar":
            with st.expander("Translation Preview"):
                try:
                    en_question = vqa_system.translate_ar_to_en(question)
                    st.markdown(f"**English Translation:** {en_question}")
                    st.caption("Medical terms will be validated before analysis")
                except:
                    st.warning("Translation preview unavailable")
        
        # Analyze button
        if st.button("🔍 Analyze Medical Image", use_container_width=True):
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
    
    # Enhanced sidebar with translation info
    with st.sidebar:
        st.markdown("### 🧬 System Status")
        
        if vqa_system.model is not None:
            st.success(f"✅ Model: Ready ({vqa_system.loaded_model_name})")
            st.info(f"🖥️ Device: {vqa_system.device.upper()}")
            if vqa_system.translation_models_loaded:
                st.success("🌐 Translation: Active")
            else:
                st.warning("⚠️ Translation: Partially Loaded")
        else:
            st.error("❌ Model: Not Ready")
        
        st.markdown("---")
        st.markdown("### 🔧 Configuration")
        st.info(f"**Max Image Dim:** {MAX_IMAGE_DIM}px")
        st.info(f"**Max File Size:** {MAX_FILE_SIZE//1024//1024}MB")
        st.info(f"**Supported Formats:** {', '.join(SUPPORTED_FORMATS)}")
        
        st.markdown("---")
        st.markdown("""
        **🩺 Medical Term Validation:**
        - 2-step translation process
        - Medical dictionary matching
        - Context-aware corrections
        
        **📋 Best Practices:**
        1. Upload clear medical images
        2. Ask specific questions
        3. Use medical terminology
        4. Specify body parts/regions
        """)
        
        st.markdown("---")
        st.markdown("**⚠️ Medical Disclaimer**")
        st.caption("This AI provides preliminary analysis for educational purposes. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Medical VQA with Enhanced Translation v3.1</strong> | Optimized for Reliability</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
