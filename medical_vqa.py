# Fixed Medical VQA Streamlit App - Working Version
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, MarianMTModel, MarianTokenizer
import logging
import time
from typing import Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_IMAGE_SIZE = (384, 384)
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
VQA_MODEL = "ButterflyCatGirl/Blip-Streamlit-chatbot"

class WorkingMedicalVQA:
    """Working Medical VQA System - Fixed Version"""
    
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
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @st.cache_resource(show_spinner=False)
    def load_models(_self):
        """Load all models - FIXED VERSION"""
        try:
            logger.info(f"Loading VQA model: {VQA_MODEL}")
            
            # Load VQA model and processor
            _self.vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL)
            _self.vqa_processor = BlipProcessor.from_pretrained(VQA_MODEL)
            _self.vqa_model = _self.vqa_model.to(_self.device)
            _self.vqa_model.eval()
            
            logger.info("Loading translation models...")
            
            # Load Arabic to English translation
            _self.ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
            _self.ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
            _self.ar_en_model = _self.ar_en_model.to(_self.device)
            _self.ar_en_model.eval()
            
            # Load English to Arabic translation
            _self.en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
            _self.en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
            _self.en_ar_model = _self.en_ar_model.to(_self.device)
            _self.en_ar_model.eval()
            
            logger.info("All models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is Arabic or English"""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_chars > 0 else "en"
    
    def translate_ar_to_en(self, text: str) -> str:
        """Translate Arabic to English using MarianMT"""
        try:
            inputs = self.ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated = self.ar_en_model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            
            return self.ar_en_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Arabic to English translation failed: {str(e)}")
            return text
    
    def translate_en_to_ar(self, text: str) -> str:
        """Translate English to Arabic using MarianMT"""
        try:
            inputs = self.en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated = self.en_ar_model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            
            return self.en_ar_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"English to Arabic translation failed: {str(e)}")
            return text
    
    def _process_image(self, image: Image.Image) -> Image.Image:
        """Process image for VQA model"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if needed
        if image.size != MAX_IMAGE_SIZE:
            image = image.resize(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        return image
    
    def process_medical_query(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Process medical VQA query - FIXED VERSION"""
        try:
            start_time = time.time()
            
            if not question.strip():
                return {"error": "Please enter a question", "success": False}
            
            # Process image
            processed_image = self._process_image(image)
            
            # Detect language and translate if needed
            detected_lang = self._detect_language(question)
            original_question = question
            
            if detected_lang == "ar":
                english_question = self.translate_ar_to_en(question)
                logger.info(f"Translated question: {english_question}")
            else:
                english_question = question
            
            # Process with VQA model - FIXED APPROACH
            inputs = self.vqa_processor(processed_image, english_question, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate answer with proper parameters
            with torch.no_grad():
                output = self.vqa_model.generate(
                    **inputs,
                    max_length=100,
                    min_length=10,
                    num_beams=5,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Decode the answer
            english_answer = self.vqa_processor.decode(output[0], skip_special_tokens=True)
            
            # Clean the answer - remove question if it appears
            if english_question.lower() in english_answer.lower():
                english_answer = english_answer.replace(english_question, "").strip()
            
            # Remove common prefixes
            prefixes_to_remove = ["answer:", "the answer is:", "response:", "result:"]
            for prefix in prefixes_to_remove:
                if english_answer.lower().startswith(prefix):
                    english_answer = english_answer[len(prefix):].strip()
            
            # Ensure we have a meaningful answer
            if len(english_answer.strip()) < 3:
                english_answer = "The medical image analysis could not provide a specific answer to this question."
            
            # Translate to Arabic
            arabic_answer = self.translate_en_to_ar(english_answer)
            
            processing_time = time.time() - start_time
            
            return {
                "original_question": original_question,
                "english_question": english_question,
                "english_answer": english_answer,
                "arabic_answer": arabic_answer,
                "detected_language": detected_lang,
                "processing_time": processing_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "error": f"Processing failed: {str(e)}",
                "success": False
            }

# Streamlit App Configuration
def init_app():
    st.set_page_config(
        page_title="Medical VQA - Fixed",
        layout="wide",
        page_icon="🩺"
    )

def apply_modern_theme():
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        .result-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #667eea;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .arabic-text {
            direction: rtl;
            text-align: right;
            font-family: 'Arial', sans-serif;
            font-size: 1.1em;
            line-height: 1.8;
            background: #f8f9ff;
            padding: 1rem;
            border-radius: 8px;
            border-right: 3px solid #764ba2;
        }
        .english-text {
            font-size: 1.1em;
            line-height: 1.8;
            background: #f8fff8;
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid #667eea;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.8rem 2rem;
            font-size: 1.1em;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .processing-stats {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #28a745;
        }
        .sidebar-content {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_vqa_system():
    return WorkingMedicalVQA()

def validate_uploaded_file(uploaded_file) -> tuple:
    if not uploaded_file:
        return False, "No file uploaded"
    
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, "File too large (max 10MB)"
    
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        return False, f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
    
    return True, "Valid file"

def main():
    init_app()
    apply_modern_theme()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Medical VQA Assistant - Fixed Version</h1>
        <p><strong>Advanced Medical Image Analysis with Accurate Bilingual Support</strong></p>
        <p>🚀 Powered by Fine-tuned BLIP + Professional Translation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize VQA system
    vqa_system = get_vqa_system()
    
    # Load models
    if vqa_system.vqa_model is None:
        with st.spinner("🔄 Loading medical AI models..."):
            success = vqa_system.load_models()
            if success:
                st.success("✅ All models loaded successfully! Ready for medical analysis!")
                st.balloons()
            else:
                st.error("❌ Failed to load models. Please check your internet connection.")
                st.stop()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Medical Image")
        
        uploaded_file = st.file_uploader(
            "Choose a medical image:",
            type=SUPPORTED_FORMATS,
            help="Upload X-ray, CT, MRI, or other medical images (max 10MB)"
        )
        
        if uploaded_file:
            is_valid, message = validate_uploaded_file(uploaded_file)
            
            if is_valid:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"📋 {uploaded_file.name}", use_container_width=True)
                    st.info(f"📊 Image size: {image.size[0]} × {image.size[1]} pixels")
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
                    uploaded_file = None
            else:
                st.error(f"❌ {message}")
                uploaded_file = None
    
    with col2:
        st.markdown("### 💬 Ask Your Medical Question")
        
        # Language selection
        language = st.selectbox(
            "🌐 Select Language:",
            options=["en", "ar"],
            format_func=lambda x: "🇺🇸 English" if x == "en" else "🇸🇦 العربية (Arabic)"
        )
        
        # Question input based on language
        if language == "ar":
            placeholder = "مثال: ما هو الجزء من الجسم الظاهر في هذه الصورة؟ هل توجد أي مشاكل؟"
            label = "🤔 السؤال الطبي:"
        else:
            placeholder = "Example: What part of the body is shown? Are there any abnormalities?"
            label = "🤔 Medical Question:"
        
        question = st.text_area(
            label,
            height=120,
            placeholder=placeholder,
            help="Ask specific questions about the medical image"
        )
        
        # Analysis button
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
                            
                            # Processing statistics
                            st.markdown(f"""
                            <div class="processing-stats">
                                ⏱️ <strong>Processing Time:</strong> {result['processing_time']:.2f} seconds | 
                                🌐 <strong>Language:</strong> {'Arabic' if result['detected_language'] == 'ar' else 'English'} | 
                                ✅ <strong>Status:</strong> Analysis Complete
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Results display
                            result_col1, result_col2 = st.columns(2)
                            
                            with result_col1:
                                st.markdown("#### 🇺🇸 English Analysis")
                                st.markdown(f"""
                                <div class="english-text">
                                    <strong>Question:</strong> {result['english_question']}<br><br>
                                    <strong>Medical Analysis:</strong> {result['english_answer']}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with result_col2:
                                st.markdown("#### 🇸🇦 التحليل الطبي بالعربية")
                                st.markdown(f"""
                                <div class="arabic-text">
                                    <strong>السؤال:</strong> {result['original_question']}<br><br>
                                    <strong>التحليل الطبي:</strong> {result['arabic_answer']}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Medical disclaimer
                            st.error("⚠️ **Medical Disclaimer: This AI analysis is for educational purposes only. Always consult qualified medical professionals for diagnosis and treatment.**")
                            st.error("⚠️ **تنبيه طبي: هذا التحليل للأغراض التعليمية فقط. استشر دائماً الأطباء المختصين للتشخيص والعلاج.**")
                            
                        else:
                            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error occurred')}")
                    
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        if vqa_system.vqa_model is not None:
            st.success("✅ VQA Model: Ready")
            st.success("✅ Translation Models: Ready") 
            st.info(f"🖥️ Device: {vqa_system.device.upper()}")
            st.info(f"🤖 Model: {VQA_MODEL}")
        else:
            st.error("❌ Models: Not Loaded")
        
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-content">
        <h4>🚀 Key Features:</h4>
        <ul>
            <li>✅ Fine-tuned medical VQA model</li>
            <li>✅ Professional translation (MarianMT)</li>
            <li>✅ Bilingual support (Arabic/English)</li>
            <li>✅ High-quality image processing</li>
            <li>✅ Optimized inference pipeline</li>
        </ul>
        
        <h4>📖 How to Use:</h4>
        <ol>
            <li>Upload medical image (X-ray, CT, MRI)</li>
            <li>Select your preferred language</li>
            <li>Ask specific medical questions</li>
            <li>Get detailed bilingual analysis</li>
        </ol>
        
        <h4>📋 Supported:</h4>
        <ul>
            <li><strong>Languages:</strong> Arabic, English</li>
            <li><strong>Formats:</strong> JPG, PNG, JPEG</li>
            <li><strong>Max Size:</strong> 10MB</li>
            <li><strong>Resolution:</strong> Auto-optimized</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚠️ Medical Disclaimer")
        st.caption("""
        This AI system provides educational analysis only. 
        It should never replace professional medical consultation. 
        Always seek qualified healthcare advice for medical decisions.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1.5rem;'>
        <p><strong>Medical VQA Assistant v3.0 - Fixed Version</strong></p>
        <p>🔬 Advanced Medical AI | 🌐 Professional Translation | ✅ Accurate Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
