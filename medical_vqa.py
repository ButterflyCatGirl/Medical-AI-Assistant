import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import io
import base64

# Configure page
st.set_page_config(
    page_title="Medical Vision AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .feature-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .result-box {
        background: #ecfdf5;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #10b981;
        margin-top: 1rem;
    }
    .error-box {
        background: #fef2f2;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ef4444;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_medical_vqa_model():
    """Load medical VQA model"""
    try:
        # Using a reliable VQA model that works well in cloud environments
        model_name = "sharawy53/final_diploma_blip-med-rad-arabic"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_name)
        st.success("✅ Medical VQA model loaded successfully!")
        return processor, model
    except Exception as e:
        st.error(f"❌ Error loading VQA model: {str(e)}")
        return None, None

@st.cache_resource
def load_translation_model():
    """Load Arabic-English translation model with proper error handling"""
    try:
        # Try the primary model first
        model_name = "Helsinki-NLP/opus-mt-ar-en"
        st.info(f"Loading translation model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Fixed: Using Seq2SeqLM
        
        st.success("✅ Translation model loaded successfully!")
        return tokenizer, model, model_name
        
    except Exception as e:
        st.warning(f"⚠️ Primary model failed: {str(e)}")
        
        # Fallback to a more reliable model
        try:
            model_name = "facebook/nllb-200-distilled-600M"
            st.info(f"Trying fallback model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            st.success("✅ Fallback translation model loaded successfully!")
            return tokenizer, model, model_name
            
        except Exception as e2:
            st.error(f"❌ All translation models failed: {str(e2)}")
            return None, None, None

def analyze_medical_image(image, question, processor, model):
    """Analyze medical image with VQA"""
    try:
        # Enhanced medical context
        medical_question = f"Medical analysis: {question}"
        
        # Process image and question
        inputs = processor(images=image, text=medical_question, return_tensors="pt", padding=True)
        
        # Generate response with better parameters
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_length=150, 
                num_beams=5,
                early_stopping=True,
                temperature=0.7
            )
        
        # Decode response
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up the response
        if generated_text.startswith("Medical analysis:"):
            generated_text = generated_text.replace("Medical analysis:", "").strip()
        
        return generated_text
        
    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}"

def translate_arabic_to_english(text, tokenizer, model, model_name):
    """Translate Arabic text to English with proper seq2seq handling"""
    try:
        # Handle different model types
        if "nllb" in model_name.lower():
            # For NLLB models, use language codes
            inputs = tokenizer(f"ara_Arab: {text}", return_tensors="pt", padding=True, truncation=True, max_length=512)
            forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    forced_bos_token_id=forced_bos_token_id
                )
        else:
            # For Helsinki-NLP models
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
        
        # Decode the translation
        translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up the translation
        if "eng_Latn:" in translated_text:
            translated_text = translated_text.replace("eng_Latn:", "").strip()
        
        return translated_text
        
    except Exception as e:
        return f"❌ Error translating text: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Vision AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", 
                                   ["Medical Image Analysis", "Arabic Translation", "About"])
    
    if app_mode == "Medical Image Analysis":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📊 Medical Image Analysis")
        st.write("Upload a medical image and ask questions about it using advanced AI.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load models
        with st.spinner("🔄 Loading AI models..."):
            vqa_processor, vqa_model = load_medical_vqa_model()
        
        if vqa_processor and vqa_model:
            # File upload
            uploaded_file = st.file_uploader("Choose a medical image...", 
                                           type=["jpg", "jpeg", "png", "bmp"])
            
            if uploaded_file is not None:
                # Display image
                image = Image.open(uploaded_file).convert("RGB")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="📸 Uploaded Medical Image", use_column_width=True)
                
                with col2:
                    # Question input with examples
                    st.write("**Example questions:**")
                    st.write("• What abnormalities do you see?")
                    st.write("• Describe the medical findings")
                    st.write("• Is this scan normal?")
                    
                    question = st.text_area("🤔 Ask a question about the medical image:", 
                                           placeholder="What abnormalities do you see in this medical image?",
                                           height=100)
                    
                    if st.button("🔍 Analyze Image", type="primary"):
                        if question:
                            with st.spinner("🧠 Analyzing medical image..."):
                                result = analyze_medical_image(image, question, vqa_processor, vqa_model)
                            
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.subheader("🔍 Analysis Result:")
                            st.write(result)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Please enter a question about the image.")
        else:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error("❌ Failed to load medical VQA models. Please refresh the page and try again.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif app_mode == "Arabic Translation":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🌐 Arabic to English Translation")
        st.write("Translate Arabic medical text to English using advanced neural translation.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load translation model
        with st.spinner("🔄 Loading translation model..."):
            translation_result = load_translation_model()
            
            if translation_result and translation_result[0] is not None:
                translation_tokenizer, translation_model, model_name = translation_result
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🇸🇦 Arabic Text")
                    arabic_text = st.text_area("Enter Arabic text:", 
                                             placeholder="أدخل النص العربي الطبي هنا...",
                                             height=200,
                                             key="arabic_input")
                
                with col2:
                    st.subheader("🇺🇸 English Translation")
                    if st.button("🔄 Translate", type="primary"):
                        if arabic_text:
                            with st.spinner("🔄 Translating..."):
                                translated_text = translate_arabic_to_english(
                                    arabic_text, 
                                    translation_tokenizer, 
                                    translation_model,
                                    model_name
                                )
                            
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.write(translated_text)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Please enter Arabic text to translate.")
                    
                    # Example translations
                    st.write("**Example Arabic medical terms:**")
                    examples = {
                        "أشعة سينية": "X-ray",
                        "فحص الدم": "Blood test", 
                        "ألم في الصدر": "Chest pain",
                        "صداع": "Headache"
                    }
                    
                    for ar, en in examples.items():
                        if st.button(f"📝 {ar}", key=f"example_{ar}"):
                            st.session_state.arabic_input = ar
                            st.experimental_rerun()
            
            else:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error("❌ Failed to load translation model. Please refresh the page and try again.")
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif app_mode == "About":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("ℹ️ About Medical Vision AI Assistant")
        st.write("""
        This application combines advanced AI technologies to assist with medical image analysis and translation:
        
        **🚀 Features:**
        - 🔍 **Medical Image Analysis**: Upload medical images (X-rays, CT scans, MRIs) and ask questions
        - 🌐 **Arabic Translation**: Translate Arabic medical text to English with high accuracy
        - 🤖 **AI-Powered**: Uses state-of-the-art vision and language models
        - 🛡️ **Reliable**: Multiple fallback models ensure consistent performance
        
        **🔧 Technologies Used:**
        - **Streamlit**: Modern web interface
        - **Transformers**: Hugging Face AI models
        - **PyTorch**: Deep learning framework
        - **BLIP**: Vision-language understanding
        - **OPUS-MT/NLLB**: Neural machine translation
        
        **📸 Supported Image Types:**
        - X-rays, CT scans, MRIs, ultrasounds
        - JPG, PNG, BMP formats
        - High resolution medical images
        
        **🌍 Language Support:**
        - Arabic to English translation
        - Medical terminology optimization
        - Context-aware translations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model status
        st.subheader("🔧 System Status")
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("Checking VQA model..."):
                vqa_proc, vqa_mod = load_medical_vqa_model()
                if vqa_proc and vqa_mod:
                    st.success("✅ VQA Model: Ready")
                else:
                    st.error("❌ VQA Model: Error")
        
        with col2:
            with st.spinner("Checking translation model..."):
                trans_result = load_translation_model()
                if trans_result and trans_result[0]:
                    st.success("✅ Translation Model: Ready")
                else:
                    st.error("❌ Translation Model: Error")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Important:** This is a demonstration application. Always consult with qualified healthcare professionals for medical decisions and diagnoses.")

if __name__ == "__main__":
    main()
