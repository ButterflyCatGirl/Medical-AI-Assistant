import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText, MarianMTModel, MarianTokenizer
from PIL import Image
import torch
import io

# Page config
st.set_page_config(
    page_title="Medical VQA - Multilingual",
    page_icon="🏥",
    layout="wide"
)

# Cache models to avoid reloading
@st.cache_resource
def load_models():
    # Load LLaVA model
    llava_model = AutoModelForImageTextToText.from_pretrained("Mohamed264/llava-medical-VQA-lora-merged3")
    processor = AutoProcessor.from_pretrained("Mohamed264/llava-medical-VQA-lora-merged3")
    
    # Load translation models
    ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    
    en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    
    return llava_model, processor, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model

# Translation functions
def translate_ar_to_en(text, ar_en_tokenizer, ar_en_model):
    inputs = ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = ar_en_model.generate(**inputs)
    return ar_en_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def translate_en_to_ar(text, en_ar_tokenizer, en_ar_model):
    inputs = en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = en_ar_model.generate(**inputs)
    return en_ar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Medical terms dictionary
medical_terms = {
    "chest x-ray": "أشعة سينية للصدر",
    "x-ray": "أشعة سينية",
    "ct scan": "تصوير مقطعي محوسب",
    "mri": "تصوير بالرنين المغناطيسي",
    "ultrasound": "تصوير بالموجات فوق الصوتية",
    "normal": "طبيعي",
    "abnormal": "غير طبيعي",
    "brain": "الدماغ",
    "fracture": "كسر",
    "no abnormality detected": "لا توجد شذوذات",
    "left lung": "الرئة اليسرى",
    "right lung": "الرئة اليمنى"
}

def translate_answer_medical(answer_en, en_ar_tokenizer, en_ar_model):
    key = answer_en.lower().strip()
    if key in medical_terms:
        return medical_terms[key]
    else:
        return translate_en_to_ar(answer_en, en_ar_tokenizer, en_ar_model)

def main():
    # Load models
    with st.spinner("🔄 Loading AI models... This may take a few minutes on first run."):
        llava_model, processor, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model = load_models()
    
    # Header
    st.title("🏥 Medical VQA - Multilingual AI Assistant")
    st.markdown("### Upload a medical image and ask questions in Arabic or English")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload a medical image (X-ray, CT, MRI, etc.)
        2. Ask your question in Arabic or English
        3. Get answers in both languages
        """)
        
        st.markdown("### 🔬 Supported Images")
        st.markdown("- Chest X-rays\n- CT Scans\n- MRI Images\n- Ultrasound\n- Other medical imaging")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🔍 Upload Medical Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a medical image for analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.markdown("#### 💬 Ask Your Question")
        question = st.text_area(
            "Enter your question (Arabic or English):",
            height=100,
            placeholder="مثال: ما هو التشخيص؟ أو What is the diagnosis?"
        )
        
        analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
    
    # Process when button clicked
    if analyze_button:
        if uploaded_file is None:
            st.error("❌ Please upload an image first!")
        elif not question.strip():
            st.error("❌ Please enter a question!")
        else:
            with st.spinner("🤖 Analyzing image and generating response..."):
                # Check if question is in Arabic
                is_arabic = any('\u0600' <= c <= '\u06FF' for c in question)
                
                if is_arabic:
                    question_ar = question.strip()
                    question_en = translate_ar_to_en(question_ar, ar_en_tokenizer, ar_en_model)
                else:
                    question_en = question.strip()
                    question_ar = translate_en_to_ar(question_en, en_ar_tokenizer, en_ar_model)
                
                # Process with LLaVA model
                inputs = processor(image, question_en, return_tensors="pt")
                with torch.no_grad():
                    output = llava_model.generate(**inputs)
                answer_en = processor.decode(output[0], skip_special_tokens=True).strip()
                
                # Translate answer
                answer_ar = translate_answer_medical(answer_en, en_ar_tokenizer, en_ar_model)
            
            # Display results
            st.markdown("## 📊 Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🟠 Arabic / العربية")
                st.info(f"**السؤال:** {question_ar}")
                st.success(f"**الإجابة:** {answer_ar}")
            
            with col2:
                st.markdown("### 🟢 English")
                st.info(f"**Question:** {question_en}")
                st.success(f"**Answer:** {answer_en}")

if __name__ == "__main__":
    main()
