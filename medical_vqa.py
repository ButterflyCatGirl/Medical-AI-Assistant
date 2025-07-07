import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import MarianTokenizer, MarianMTModel
import logging
import time
import gc

# Suppress warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_IMAGE_SIZE = (512, 512)
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MODEL_NAME = "ButterflyCatGirl/Blip-Streamlit-chatbot"

# Translation models
ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

class MedicalVQA:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"Loading model: {MODEL_NAME}")
            self.processor = BlipProcessor.from_pretrained(MODEL_NAME)
            self.model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

    def translate_ar_to_en(self, text):
        inputs = ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = ar_en_model.generate(**inputs)
        return ar_en_tokenizer.decode(translated[0], skip_special_tokens=True)

    def translate_en_to_ar(self, text):
        inputs = en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = en_ar_model.generate(**inputs)
        return en_ar_tokenizer.decode(translated[0], skip_special_tokens=True)

    def process_image(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image.resize(MAX_IMAGE_SIZE, resample=Image.BILINEAR)

    def analyze(self, image, question):
        try:
            start_time = time.time()
            
            # Translate question if Arabic
            detected_lang = "ar" if any(c in question for c in ['أ', 'ب', 'ت']) else "en"
            if detected_lang == "ar":
                english_question = self.translate_ar_to_en(question)
            else:
                english_question = question

            # Process image
            image = self.process_image(image)
            
            # Generate answer
            inputs = self.processor(image, english_question, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(**inputs)
            
            english_answer = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Translate back to Arabic if needed
            arabic_answer = self.translate_en_to_ar(english_answer) if detected_lang == "ar" else ""
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "answer_en": english_answer,
                "answer_ar": arabic_answer,
                "detected_lang": detected_lang,
                "time": processing_time
            }
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}

def main():
    st.set_page_config(
        page_title="🩺 Medical VQA",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
    <style>
        .result-box { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1em 0; }
        .arabic-text { direction: rtl; text-align: right; font-family: Arial; }
    </style>
    """, unsafe_allow_html=True)

    vqa = MedicalVQA()
    if not vqa.model:
        st.error("❌ Model loading failed")
        return

    uploaded_file = st.file_uploader("Upload X-ray/MRI/CT Scan", type=SUPPORTED_FORMATS)
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            question = st.text_area("Medical Question (Arabic/English)", height=100)
            if st.button("🔍 Accurate Analysis"):
                if not question.strip():
                    st.warning("Enter a question")
                    return
                
                with st.spinner("Analyzing..."):
                    result = vqa.analyze(image, question)
                    
                    if result["success"]:
                        st.markdown("---")
                        st.markdown(f"⏱️ Analysis Time: {result['time']:.2f}s")
                        
                        st.markdown("### 🧾 Results")
                        st.markdown(f"**Question**: {question}")
                        
                        st.markdown("#### 🇺🇸 English Analysis")
                        st.markdown(f"**Finding**: {result['answer_en']}")
                        
                        if result["detected_lang"] == "ar":
                            st.markdown("#### 🇪🇬 التحليل الطبي بالعربية")
                            st.markdown(f"""
                            <div class="arabic-text">
                                <strong>النتيجة:</strong> {result['answer_ar']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.warning("⚠️ For educational purposes only - consult a specialist")
                    else:
                        st.error(f"❌ Analysis failed: {result['error']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
