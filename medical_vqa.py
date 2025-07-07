
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging
import time
import gc
import warnings
import re
from typing import Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_IMAGE_SIZE = (256, 256)  # Faster processing
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
FINE_TUNED_MODEL = "sharawy53/blip-vqa-medical-arabic"

class MedicalVQAOptimized:
    """Optimized medical VQA system"""
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.medical_terms = self._load_medical_terms()

    def _load_medical_terms(self) -> Dict[str, str]:
        """Essential medical terms for translation"""
        return {
            "normal": "طبيعي", "abnormal": "غير طبيعي", "disease": "مرض",
            "heart": "القلب", "lung": "الرئة", "lungs": "الرئتان",
            "pneumonia": "التهاب رئوي", "fracture": "كسر", "tumor": "ورم"
        }

    @st.cache_resource
    def load_model(_self):
        """Load model with optimizations"""
        try:
            logger.info(f"Loading model: {FINE_TUNED_MODEL}")
            _self.processor = BlipProcessor.from_pretrained(FINE_TUNED_MODEL)
            _self.model = BlipForConditionalGeneration.from_pretrained(
                FINE_TUNED_MODEL,
                torch_dtype=torch.float16 if _self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(_self.device)
            _self.model.eval()
            return True
        except Exception as e:
            logger.error(f"Model load error: {str(e)}")
            return False

    def _detect_language(self, text: str) -> str:
        """Detect Arabic/English"""
        return "ar" if any('\u0600' <= c <= '\u06FF' for c in text) else "en"

    def _translate_medical(self, text: str) -> str:
        """Fast medical term translation"""
        words = text.split()
        translated = []
        for word in words:
            clean = re.sub(r'[^\w]', '', word.lower())
            translated.append(self.medical_terms.get(clean, word))
        return " ".join(translated)

    def _process_image(self, image: Image.Image) -> Image.Image:
        """Fast image preprocessing"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image.resize(MAX_IMAGE_SIZE, Image.BILINEAR)

    def analyze(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Analyze image with optimized settings"""
        try:
            start = time.time()
            image = self._process_image(image)
            
            inputs = self.processor(
                image, question, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=80,
                    num_beams=3,
                    early_stopping=True,
                    temperature=0.7,
                    top_p=0.9
                )

            answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            answer = re.sub(r'^[,\.\:\?\!]+\s*', '', answer).strip()
            is_ar = self._detect_language(question)

            # Clean up memory
            del inputs, generated_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            return {
                "success": True,
                "en": answer or "No clear findings detected",
                "ar": self._translate_medical(answer) if is_ar else "",
                "time": round(time.time() - start, 2)
            }
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}

# Streamlit App
def main():
    st.set_page_config(
        page_title="🩺 Medical VQA",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # CSS
    st.markdown("""
    <style>
        .header {color: #1E90FF; font-size: 1.5em; margin-bottom: 1em;}
        .result {padding: 1em; background: #f8f9fa; border-radius: 8px; margin: 1em 0;}
        .arabic {direction: rtl; text-align: right; font-family: Arial;}
        .small {font-size: 0.9em; color: #666;}
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<div class='header'>🩺 Medical Image Analysis (AI)</div>", unsafe_allow_html=True)

    # Initialize model
    vqa = MedicalVQAOptimized()
    if not vqa.load_model():
        st.error("❌ Model failed to load")
        return

    # Upload
    uploaded = st.file_uploader("Upload X-ray/MRI/CT Scan", type=SUPPORTED_FORMATS)
    if uploaded:
        try:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Question
            lang = st.radio("Language", ["English", "Arabic"], horizontal=True)
            question = st.text_input(
                "Medical Question",
                value="What is abnormal in this image?" if lang == "English" else "ما غير طبيعي في هذه الصورة؟"
            )

            if st.button("🔍 Analyze"):
                if not question.strip():
                    st.warning("Enter a question")
                    return
                
                with st.spinner("Analyzing..."):
                    result = vqa.analyze(image, question)
                    
                    if result["success"]:
                        st.markdown("---")
                        st.markdown(f"⏱️ Analysis Time: {result['time']} seconds")
                        st.markdown("### 🧾 Results")
                        st.markdown(f"**Question**: {question}")
                        
                        # English result
                        st.markdown(f"**Finding (English)**: {result['en']}")
                        
                        # Arabic result
                        if lang == "Arabic":
                            st.markdown(f"<div class='arabic'>**النتيجة الطبية**: {result['ar']}</div>", 
                                      unsafe_allow_html=True)
                        
                        st.warning("⚠️ For educational purposes only - consult a specialist for diagnosis")
                    else:
                        st.error(f"❌ Analysis failed: {result['error']}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Sidebar
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This AI provides preliminary medical image analysis.
    - Supports: X-rays, CT scans, MRI
    - Works best for: Chest, lungs, bones
    - Uses model: sharawy53/blip-vqa-medical-arabic
    
    ⚠️ **Not a substitute for professional diagnosis**
    """)

if __name__ == "__main__":
    main()
