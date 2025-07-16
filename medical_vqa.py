import streamlit as st
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoModelForVision2Seq,
    MarianMTModel,
    MarianTokenizer
)
from PIL import Image
import gc
import warnings
warnings.filterwarnings("ignore")

# Configure page
st.set_page_config(
    page_title="Medical AI Assistant",
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
        background: linear-gradient(90deg, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .feature-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .result-box {
        background: #ecfdf5;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #10b981;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .error-box {
        background: #fef2f2;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ef4444;
        color: #dc2626;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #10b981);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=True)
def load_medical_vqa_model():
    """Load your fine-tuned medical VQA model"""
    try:
        # Your actual fine-tuned model
        model_name = "sharawy53/final_diploma_blip-med-rad-arabic"
        
        # Try to load your model, fallback to working alternative
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            st.success("✅ Successfully loaded your fine-tuned medical VQA model!")
            return processor, model, "custom"
        except Exception as e:
            st.warning(f"⚠️ Could not load custom model: {str(e)}")
            st.info("🔄 Loading fallback medical vision model...")
            
            # Fallback to a working vision model
            fallback_model = "microsoft/git-base-coco"
            processor = AutoProcessor.from_pretrained(fallback_model)
            model = AutoModelForVision2Seq.from_pretrained(
                fallback_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            return processor, model, "fallback"
            
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None

@st.cache_resource(show_spinner=True)
def load_translation_model():
    """Load Arabic-English translation model"""
    try:
        model_name = "Helsinki-NLP/opus-mt-ar-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading translation model: {str(e)}")
        return None, None

def analyze_medical_image(image, question, processor, model, model_type):
    """Analyze medical image with VQA"""
    try:
        if model_type == "custom":
            # For your fine-tuned model
            prompt = f"<image>\nHuman: {question}\nAssistant:"
            inputs = processor(prompt, image, return_tensors="pt")
        else:
            # For fallback model
            inputs = processor(images=image, text=question, return_tensors="pt", padding=True)
        
        # Generate response
        with torch.no_grad():
            if model_type == "custom":
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                # Decode only the new tokens
                response = processor.tokenizer.decode(
                    generated_ids[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
            else:
                generated_ids = model.generate(**inputs, max_length=100, num_beams=4)
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up memory
        del inputs, generated_ids
        gc.collect()
        
        return response
        
    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}"

def translate_arabic_to_english(text, tokenizer, model):
    """Translate Arabic text to English"""
    try:
        # Prepare input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate translation
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode translation
        translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up memory
        del inputs, generated_ids
        gc.collect()
        
        return translated_text
        
    except Exception as e:
        return f"❌ Error translating text: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by Fine-tuned Medical VQA Model & Arabic Translation**")
    
    # Sidebar
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.selectbox(
        "Choose Mode:", 
        ["🔍 Medical Image Analysis", "🌐 Arabic Translation", "ℹ️ About"],
        index=0
    )
    
    # Model loading status
    with st.sidebar:
        st.markdown("### 🤖 Model Status")
        model_status = st.empty()
    
    if app_mode == "🔍 Medical Image Analysis":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📊 Medical Image Analysis")
        st.write("Upload medical images (X-rays, CT scans, MRIs) and ask specific questions about them.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load models
        with st.spinner("🔄 Loading medical AI models..."):
            vqa_processor, vqa_model, model_type = load_medical_vqa_model()
            
        if vqa_processor and vqa_model:
            model_status.success(f"✅ VQA Model Ready ({model_type})")
            
            # File upload
            uploaded_file = st.file_uploader(
                "📎 Choose a medical image...", 
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                help="Supported formats: JPG, PNG, BMP, TIFF"
            )
            
            if uploaded_file is not None:
                # Display image
                image = Image.open(uploaded_file).convert("RGB")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="📋 Uploaded Medical Image", use_container_width=True)
                    st.info(f"📏 Image size: {image.size[0]}x{image.size[1]} pixels")
                
                with col2:
                    # Predefined questions for medical images
                    st.markdown("#### 🎯 Quick Questions:")
                    quick_questions = [
                        "What abnormalities do you see in this medical image?",
                        "Describe the main findings in this image.",
                        "What anatomical structures are visible?",
                        "Are there any signs of pathology?",
                        "What type of medical imaging is this?"
                    ]
                    
                    selected_question = st.selectbox("Select a question:", ["Custom question..."] + quick_questions)
                    
                    # Question input
                    if selected_question == "Custom question...":
                        question = st.text_area(
                            "💭 Ask a question about the medical image:", 
                            placeholder="What abnormalities do you see in this X-ray?",
                            height=100
                        )
                    else:
                        question = selected_question
                        st.text_area("Selected question:", value=question, height=100, disabled=True)
                    
                    if st.button("🔍 Analyze Image", type="primary"):
                        if question:
                            with st.spinner("🧠 AI is analyzing the medical image..."):
                                result = analyze_medical_image(image, question, vqa_processor, vqa_model, model_type)
                            
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.subheader("🔍 Medical Analysis Result:")
                            st.markdown(f"**Question:** {question}")
                            st.markdown(f"**Answer:** {result}")
                            
                            # Add disclaimer
                            st.markdown("---")
                            st.warning("⚠️ **Medical Disclaimer:** This AI analysis is for educational purposes only. Always consult healthcare professionals for medical decisions.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Please enter a question about the image.")
            else:
                st.info("👆 Please upload a medical image to begin analysis.")
        else:
            model_status.error("❌ Model Loading Failed")
            st.error("❌ Failed to load medical VQA models. Please refresh the page to try again.")
    
    elif app_mode == "🌐 Arabic Translation":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🌐 Arabic to English Translation")
        st.write("Translate medical terminology and text from Arabic to English.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load translation model
        with st.spinner("🔄 Loading translation model..."):
            translation_tokenizer, translation_model = load_translation_model()
        
        if translation_tokenizer and translation_model:
            model_status.success("✅ Translation Model Ready")
            
            # Sample medical terms
            st.markdown("#### 🏥 Sample Medical Terms:")
            sample_terms = {
                "صداع": "Headache",
                "ألم في الصدر": "Chest pain", 
                "ضغط الدم": "Blood pressure",
                "مرض السكري": "Diabetes",
                "التهاب المفاصل": "Arthritis"
            }
            
            cols = st.columns(len(sample_terms))
            for i, (arabic, english) in enumerate(sample_terms.items()):
                with cols[i]:
                    if st.button(f"{arabic}\n({english})", key=f"sample_{i}"):
                        st.session_state.arabic_input = arabic
            
            # Text input
            arabic_text = st.text_area(
                "📝 Enter Arabic text:", 
                value=st.session_state.get('arabic_input', ''),
                placeholder="أدخل النص العربي الطبي هنا...",
                height=150,
                help="Enter medical text in Arabic for translation"
            )
            
            if st.button("🌐 Translate", type="primary"):
                if arabic_text.strip():
                    with st.spinner("🔄 Translating Arabic text..."):
                        translated_text = translate_arabic_to_english(arabic_text, translation_tokenizer, translation_model)
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("📝 Translation Result:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Arabic Text:**")
                        st.text_area("", value=arabic_text, height=100, disabled=True)
                    
                    with col2:
                        st.markdown("**English Translation:**")
                        st.text_area("", value=translated_text, height=100, disabled=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please enter Arabic text to translate.")
        else:
            model_status.error("❌ Translation Model Failed")
            st.error("❌ Failed to load translation model. Please refresh the page to try again.")
    
    elif app_mode == "ℹ️ About":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("ℹ️ About Medical AI Assistant")
        
        st.markdown("""
        ### 🎯 Purpose
        This application combines advanced AI technologies to assist healthcare professionals and students with medical image analysis and Arabic-English translation.
        
        ### ✨ Key Features
        - **🔍 Medical Image Analysis**: 
          - Upload X-rays, CT scans, MRIs, and other medical images
          - Ask specific questions about findings and abnormalities
          - Get AI-powered insights using fine-tuned medical models
        
        - **🌐 Arabic Translation**: 
          - Translate medical terminology from Arabic to English
          - Support for complex medical phrases and descriptions
          - Built for healthcare communication
        
        ### 🛠️ Technology Stack
        - **Frontend**: Streamlit
        - **AI Models**: 
          - Custom fine-tuned LLaVA medical VQA model
          - Helsinki NLP Arabic-English translation
        - **Backend**: PyTorch, Transformers
        - **Image Processing**: PIL, OpenCV
        
        ### 📊 Model Information
        - **Medical VQA**: `Mohamed264/llava-medical-VQA-lora-merged3`
        - **Translation**: `Helsinki-NLP/opus-mt-ar-en`
        - **Optimization**: Memory-efficient inference with automatic cleanup
        
        ### 🎓 Use Cases
        - Medical education and training
        - Radiology assistance
        - Medical documentation translation
        - Healthcare communication support
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance info
        with st.expander("⚡ Performance & Technical Details"):
            st.markdown("""
            - **Memory Management**: Automatic garbage collection after each inference
            - **GPU Support**: Automatic detection and utilization when available
            - **Model Caching**: Efficient loading with Streamlit caching
            - **Error Handling**: Graceful fallbacks for model loading issues
            - **Response Time**: 2-10 seconds depending on image size and question complexity
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 1rem;'>
        💡 <strong>Medical Disclaimer:</strong> This AI assistant is for educational and research purposes only. 
        Always consult qualified healthcare professionals for medical diagnoses and treatment decisions.
        <br><br>
        🚀 Built with Streamlit | 🤖 Powered by Transformers | 💻 Deployed on Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
