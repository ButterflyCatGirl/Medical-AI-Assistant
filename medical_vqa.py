import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, MarianTokenizer, MarianMTModel
from PIL import Image
import requests
import io
import time
from typing import Dict, List, Tuple, Optional
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Medical VQA Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #DCF8C6;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background-color: #F1F1F1;
        margin-right: auto;
    }
    .arabic-text {
        font-family: 'Arial', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stButton > button {
        width: 100%;
        background-color: #2E86AB;
        color: white;
    }
    .upload-section {
        border: 2px dashed #2E86AB;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MedicalVQAChatbot:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False
        self.load_models()
    
    @st.cache_resource
    def load_models(_self):
        """Load all required models with caching"""
        try:
            with st.spinner("Loading Medical VQA Model... This may take a few minutes on first run."):
                # Load VQA model
                _self.vqa_processor = AutoProcessor.from_pretrained("Mohamed264/llava-medical-VQA-lora-merged3")
                _self.vqa_model = AutoModelForImageTextToText.from_pretrained(
                    "Mohamed264/llava-medical-VQA-lora-merged3",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            with st.spinner("Loading Translation Models..."):
                # Load translation models
                _self.en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
                _self.en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
                
                _self.ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
                _self.ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
            
            _self.models_loaded = True
            st.success("✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Arabic or English"""
        arabic_chars = 0
        english_chars = 0
        
        for char in text:
            if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
                arabic_chars += 1
            elif 'a' <= char.lower() <= 'z':
                english_chars += 1
        
        return 'ar' if arabic_chars > english_chars else 'en'
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between Arabic and English"""
        if source_lang == target_lang:
            return text
        
        try:
            if source_lang == 'ar' and target_lang == 'en':
                inputs = self.ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    translated = self.ar_en_model.generate(**inputs, max_length=512, num_beams=4)
                translated_text = self.ar_en_tokenizer.decode(translated[0], skip_special_tokens=True)
                
            elif source_lang == 'en' and target_lang == 'ar':
                inputs = self.en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    translated = self.en_ar_model.generate(**inputs, max_length=512, num_beams=4)
                translated_text = self.en_ar_tokenizer.decode(translated[0], skip_special_tokens=True)
            else:
                return text
                
            return translated_text.strip()
            
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return text
    
    def process_medical_vqa(self, image: Image.Image, question: str) -> Tuple[str, str]:
        """Process medical VQA with image and question"""
        try:
            # Detect input language
            input_lang = self.detect_language(question)
            
            # Translate question to English if needed
            english_question = question
            if input_lang == 'ar':
                english_question = self.translate_text(question, 'ar', 'en')
            
            # Prepare the prompt for LLaVA
            prompt = f"USER: <image>\n{english_question}\nASSISTANT:"
            
            # Process inputs
            inputs = self.vqa_processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.vqa_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.vqa_processor.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.vqa_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "ASSISTANT:" in full_response:
                english_response = full_response.split("ASSISTANT:")[-1].strip()
            else:
                english_response = full_response.strip()
            
            # Clean up the response
            english_response = english_response.replace("USER:", "").replace("<image>", "").strip()
            
            # Translate response back to user's language if needed
            final_response = english_response
            if input_lang == 'ar' and english_response:
                final_response = self.translate_text(english_response, 'en', 'ar')
            
            return final_response, input_lang
            
        except Exception as e:
            error_msg = f"Error processing VQA: {str(e)}"
            st.error(error_msg)
            return error_msg, 'en'

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MedicalVQAChatbot()
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

def display_chat_message(message: Dict, is_user: bool = False):
    """Display a chat message with proper styling"""
    css_class = "user-message" if is_user else "bot-message"
    lang_class = "arabic-text" if message.get('language') == 'ar' else ""
    
    st.markdown(f"""
    <div class="chat-message {css_class} {lang_class}">
        <strong>{'You' if is_user else 'Medical Assistant'}:</strong><br>
        {message['content']}
        <br><small>{message['timestamp']}</small>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown("<h1 class='main-header'>🏥 Medical VQA Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Upload medical images and ask questions in English or Arabic | ارفع الصور الطبية واسأل بالإنجليزية أو العربية</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions | التعليمات")
        st.markdown("""
        **English:**
        1. Upload a medical image
        2. Ask your question
        3. Get AI-powered analysis
        
        **العربية:**
        1. ارفع صورة طبية
        2. اسأل سؤالك
        3. احصل على تحليل بالذكاء الاصطناعي
        """)
        
        st.header("🔧 System Status")
        if st.session_state.chatbot.models_loaded:
            st.success("✅ Models Ready")
            st.info(f"Device: {st.session_state.chatbot.device}")
        else:
            st.error("❌ Models Not Loaded")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat | مسح المحادثة"):
            st.session_state.chat_history = []
            st.session_state.uploaded_image = None
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat history display
        st.subheader("💬 Chat History | سجل المحادثة")
        
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    display_chat_message(message, message['sender'] == 'user')
            else:
                st.info("Start a conversation by uploading an image and asking a question!")
    
    with col2:
        # Image upload section
        st.subheader("📤 Upload Image | رفع الصورة")
        
        uploaded_file = st.file_uploader(
            "Choose a medical image | اختر صورة طبية",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image | الصورة المرفوعة", use_column_width=True)
                st.session_state.uploaded_image = image
                st.success("✅ Image uploaded successfully! | تم رفع الصورة بنجاح!")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    # Question input section
    st.subheader("❓ Ask Your Question | اطرح سؤالك")
    
    # Create columns for question input and send button
    col_q1, col_q2 = st.columns([4, 1])
    
    with col_q1:
        user_question = st.text_area(
            "Enter your question in English or Arabic:",
            placeholder="What abnormality is visible in this X-ray? | ما هو الشذوذ المرئي في هذه الأشعة السينية؟",
            height=100,
            key="question_input"
        )
    
    with col_q2:
        send_button = st.button("🚀 Send | إرسال", type="primary", use_container_width=True)
    
    # Process question when send button is clicked
    if send_button and user_question.strip():
        if st.session_state.uploaded_image is None:
            st.warning("⚠️ Please upload an image first! | يرجى رفع صورة أولاً!")
        elif not st.session_state.chatbot.models_loaded:
            st.error("❌ Models are not loaded. Please refresh the page. | النماذج غير محملة. يرجى تحديث الصفحة.")
        else:
            # Show processing message
            with st.spinner("Processing your question... | جاري معالجة سؤالك..."):
                # Add user message to chat history
                user_message = {
                    'content': user_question,
                    'sender': 'user',
                    'timestamp': time.strftime("%H:%M:%S"),
                    'language': st.session_state.chatbot.detect_language(user_question)
                }
                st.session_state.chat_history.append(user_message)
                
                # Process VQA
                response, detected_lang = st.session_state.chatbot.process_medical_vqa(
                    st.session_state.uploaded_image,
                    user_question
                )
                
                # Add bot response to chat history
                bot_message = {
                    'content': response,
                    'sender': 'bot',
                    'timestamp': time.strftime("%H:%M:%S"),
                    'language': detected_lang
                }
                st.session_state.chat_history.append(bot_message)
                
                # Clear the input
                st.session_state.question_input = ""
                
            # Rerun to update the chat display
            st.rerun()
    
    # Example questions section
    st.subheader("💡 Example Questions | أسئلة نموذجية")
    
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.markdown("""
        **English Examples:**
        - What abnormality is visible in this X-ray?
        - Describe the pathological findings in this image
        - What is the diagnosis based on this medical image?
        - Are there any signs of infection or inflammation?
        """)
    
    with col_ex2:
        st.markdown("""
        **أمثلة بالعربية:**
        - ما هو الشذوذ المرئي في هذه الأشعة السينية؟
        - صف النتائج المرضية في هذه الصورة
        - ما هو التشخيص بناءً على هذه الصورة الطبية؟
        - هل هناك علامات عدوى أو التهاب؟
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🏥 Medical VQA Chatbot | Powered by LLaVA Medical VQA Model<br>
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. Always consult healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
