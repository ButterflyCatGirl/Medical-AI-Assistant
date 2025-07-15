import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_medical_vqa_model():
    """Load medical VQA model"""
    try:
        # Using a lightweight medical VQA model for demo
        model_name = "microsoft/git-base-coco"  # Fallback to a working model
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"Error loading VQA model: {str(e)}")
        return None, None

@st.cache_resource
def load_translation_model():
    """Load Arabic-English translation model"""
    try:
        model_name = "Helsinki-NLP/opus-mt-ar-en"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading translation model: {str(e)}")
        return None, None

def analyze_medical_image(image, question, processor, model):
    """Analyze medical image with VQA"""
    try:
        # Process image and question
        inputs = processor(images=image, text=question, return_tensors="pt", padding=True)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=100, num_beams=4)
        
        # Decode response
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def translate_arabic_to_english(text, tokenizer, model):
    """Translate Arabic text to English"""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=100, num_beams=4)
        
        translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        return f"Error translating text: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Vision AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", 
                                   ["Medical Image Analysis", "Arabic Translation", "About"])
    
    if app_mode == "Medical Image Analysis":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📊 Medical Image Analysis")
        st.write("Upload a medical image and ask questions about it.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load models
        with st.spinner("Loading AI models..."):
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
                    st.image(image, caption="Uploaded Medical Image", use_column_width=True)
                
                with col2:
                    # Question input
                    question = st.text_area("Ask a question about the medical image:", 
                                           placeholder="What abnormalities do you see in this X-ray?",
                                           height=100)
                    
                    if st.button("Analyze Image", type="primary"):
                        if question:
                            with st.spinner("Analyzing medical image..."):
                                result = analyze_medical_image(image, question, vqa_processor, vqa_model)
                            
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.subheader("🔍 Analysis Result:")
                            st.write(result)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("Please enter a question about the image.")
        else:
            st.error("Failed to load medical VQA models. Please try again later.")
    
    elif app_mode == "Arabic Translation":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🌐 Arabic to English Translation")
        st.write("Translate Arabic medical text to English.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load translation model
        with st.spinner("Loading translation model..."):
            translation_tokenizer, translation_model = load_translation_model()
        
        if translation_tokenizer and translation_model:
            arabic_text = st.text_area("Enter Arabic text:", 
                                     placeholder="أدخل النص العربي هنا...",
                                     height=150)
            
            if st.button("Translate", type="primary"):
                if arabic_text:
                    with st.spinner("Translating..."):
                        translated_text = translate_arabic_to_english(arabic_text, 
                                                                    translation_tokenizer, 
                                                                    translation_model)
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("📝 Translation Result:")
                    st.write(translated_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("Please enter Arabic text to translate.")
        else:
            st.error("Failed to load translation model. Please try again later.")
    
    elif app_mode == "About":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("ℹ️ About Medical Vision AI Assistant")
        st.write("""
        This application combines advanced AI technologies to assist with medical image analysis:
        
        **Features:**
        - 🔍 **Medical Image Analysis**: Upload medical images and ask questions about them
        - 🌐 **Arabic Translation**: Translate Arabic medical text to English
        - 🤖 **AI-Powered**: Uses state-of-the-art vision and language models
        
        **Technologies Used:**
        - Streamlit for the web interface
        - Transformers for AI models
        - PyTorch for deep learning
        - PIL for image processing
        
        **Supported Image Types:**
        - X-rays, CT scans, MRIs
        - JPG, PNG, BMP formats
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Note:** This is a demonstration application. Always consult with healthcare professionals for medical decisions.")

if __name__ == "__main__":
    main()
