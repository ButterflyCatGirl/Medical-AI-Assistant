# Ultra-Fast Medical VQA Streamlit App with Accurate Translation
import streamlit as st
from PIL import Image, ImageOps
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, MarianMTModel, MarianTokenizer
import logging
import time
import gc
from typing import Optional, Dict, Any
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized Configuration
MAX_IMAGE_DIM = 512  # Higher resolution for medical details
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
BASE_MODEL = "Salesforce/blip-vqa-base"  # Official base model

class AccurateMedicalVQA:
    """Accurate Medical VQA System with Enhanced Translation"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = self._get_device()
        self.translation_models_loaded = False
        
    def _get_device(self) -> str:
        """Get optimal device"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _load_translation_models(self):
        """Load translation models"""
        try:
            logger.info("Loading translation models...")
            # Arabic to English translation
            self.ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
            self.ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en").to(self.device)
            
            # English to Arabic translation
            self.en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
            self.en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar").to(self.device)
            
            logger.info("Translation models loaded successfully")
            self.translation_models_loaded = True
            return True
        except Exception as e:
            logger.error(f"Translation model loading failed: {str(e)}")
            return False
    
    def translate_ar_to_en(self, text: str) -> str:
        """Translate Arabic to English"""
        if not text.strip():
            return ""
        
        if not self.translation_models_loaded:
            self._load_translation_models()
        
        try:
            inputs = self.ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.ar_en_model.generate(**inputs)
            return self.ar_en_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Arabic to English translation failed: {str(e)}")
            return text
    
    def translate_en_to_ar(self, text: str) -> str:
        """Translate English to Arabic"""
        if not text.strip():
            return ""
        
        if not self.translation_models_loaded:
            self._load_translation_models()
        
        try:
            inputs = self.en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.en_ar_model.generate(**inputs)
            return self.en_ar_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"English to Arabic translation failed: {str(e)}")
            return text
    
    @st.cache_resource(show_spinner=False)
    def load_model(_self):
        """Load model with robust error handling"""
        try:
            logger.info(f"Loading model: {BASE_MODEL}")
            
            # Load processor and model
            _self.processor = BlipProcessor.from_pretrained(BASE_MODEL)
            
            # Handle device and precision
            if _self.device == "cpu":
                _self.model = BlipForQuestionAnswering.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch.float32
                )
            else:
                _self.model = BlipForQuestionAnswering.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch.float16
                )
            
            _self.model = _self.model.to(_self.device)
            _self.model.eval()
            
            logger.info(f"Model loaded successfully on {_self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            # Try alternative loading approach
            try:
                logger.info("Trying alternative model loading approach...")
                _self.processor = BlipProcessor.from_pretrained(BASE_MODEL)
                _self.model = BlipForQuestionAnswering.from_pretrained(BASE_MODEL)
                _self.model = _self.model.to(_self.device)
                _self.model.eval()
                logger.info("Model loaded successfully with alternative approach")
                return True
            except Exception as alt_e:
                logger.error(f"Alternative loading failed: {str(alt_e)}")
                return False
    
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
        """Medical-aware cleaning of
