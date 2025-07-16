import streamlit as st
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import re
import time
from deep_translator import GoogleTranslator

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
    /* ... (أبقى نفس تنسيقات CSS السابقة بدون تغيير) ... */
    .translation-box div {
        margin: 0.5rem 0;
        padding: 0.5rem;
        border-radius: 0.25rem;
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ... (أبقى نفس الدوال السابقة بدون تغيير) ...

def main():
    # ... (أبقى نفس تهيئة الحالة السابقة) ...
    
    if app_mode == "Medical Image Analysis":
        # ... (أبقى نفس كود تحميل النموذج السابق) ...
        
        if vqa_processor and vqa_model:
            # ... (أبقى نفس كود تحميل الصورة السابق) ...
                    
                    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                        if question:
                            # ... (أبقى نفس كود الترجمة السابق) ...
                            
                            # Display results
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.subheader("🔍 Analysis Result")
                            
                            # Display question in both languages - FIXED ARABIC DISPLAY
                            st.markdown("""
                            <div class="translation-box">
                                <div>
                                    <strong>Your Question:</strong> 
                                    <span>{display_question_en}</span>
                                    <span class="language-badge english-badge">EN</span>
                                    {auto_en}
                                </div>
                                <div>
                                    <strong>سؤالك:</strong> 
                                    <span>{display_question_ar}</span>
                                    <span class="language-badge arabic-badge">AR</span>
                                    {auto_ar}
                                </div>
                            </div>
                            """.format(
                                display_question_en=display_question_en,
                                display_question_ar=display_question_ar,
                                auto_en="<span class='warning-badge'>Auto</span>" if "[Auto]" in display_question_en else "",
                                auto_ar="<span class='warning-badge'>Auto</span>" if "[Auto]" in display_question_ar else ""
                            ), unsafe_allow_html=True)
                            
                            # Display answer in both languages
                            st.markdown("""
                            <div class="translation-box">
                                <div>
                                    <strong>Answer:</strong> 
                                    <span>{english_answer}</span>
                                    <span class="language-badge english-badge">EN</span>
                                    {auto_ans}
                                </div>
                                <div>
                                    <strong>الإجابة:</strong> 
                                    <span>{arabic_answer}</span>
                                    <span class="language-badge arabic-badge">AR</span>
                                </div>
                            </div>
                            """.format(
                                english_answer=english_answer,
                                arabic_answer=arabic_answer,
                                auto_ans="<span class='warning-badge'>Auto</span>" if "[Auto]" in english_answer else ""
                            ), unsafe_allow_html=True)
                            
                            # Add confidence disclaimer
                            st.caption("⚠️ **Medical AI Disclaimer**: This analysis is for educational purposes only. Always consult healthcare professionals for medical decisions.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("Please enter a question about the image.")
    # ... (أبقى باقي الكود كما هو) ...

if __name__ == "__main__":
    main()
