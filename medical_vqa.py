import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    MarianMTModel,
    MarianTokenizer,
)

# ---------------------------------------------------------
# 1) Load every heavy resource exactly **once**
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_nlp_models():
    # BLIP VQA
    blip_model = BlipForQuestionAnswering.from_pretrained("sharawy53/diploma")
    blip_processor = BlipProcessor.from_pretrained("sharawy53/diploma")

    # Translation models
    ar_en_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

    en_ar_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

    return {
        "blip_model": blip_model,
        "blip_processor": blip_processor,
        "ar_en_tok": ar_en_tok,
        "ar_en_model": ar_en_model,
        "en_ar_tok": en_ar_tok,
        "en_ar_model": en_ar_model,
    }


models = load_nlp_models()

# ---------------------------------------------------------
# 2) Helper: translation
# ---------------------------------------------------------
def translate_ar_to_en(text: str) -> str:
    tkn = models["ar_en_tok"]
    mdl = models["ar_en_model"]
    inputs = tkn(text, return_tensors="pt", padding=True, truncation=True)
    out = mdl.generate(**inputs)
    return tkn.decode(out[0], skip_special_tokens=True).strip()


def translate_en_to_ar(text: str) -> str:
    tkn = models["en_ar_tok"]
    mdl = models["en_ar_model"]
    inputs = tkn(text, return_tensors="pt", padding=True, truncation=True)
    out = mdl.generate(**inputs)
    return tkn.decode(out[0], skip_special_tokens=True).strip()


# “Smart” manual medical dictionary
MED_DICT = {
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
    "right lung": "الرئة اليمنى",
}


def smart_translate_answer(answer_en: str) -> str:
    key = answer_en.lower().strip()
    return MED_DICT.get(key, translate_en_to_ar(answer_en))


# ---------------------------------------------------------
# 3) Core VQA pipeline
# ---------------------------------------------------------
def vqa_multilingual(image: Image.Image, question: str):
    if image is None or question.strip() == "":
        return "", "", "", ""

    # Detect script
    is_arabic = any("\u0600" <= c <= "\u06FF" for c in question)

    if is_arabic:
        q_ar = question.strip()
        q_en = translate_ar_to_en(q_ar)
    else:
        q_en = question.strip()
        q_ar = translate_en_to_ar(q_en)

    proc = models["blip_processor"]
    blip = models["blip_model"]

    inputs = proc(image, q_en, return_tensors="pt")
    with torch.no_grad():
        output = blip.generate(**inputs)
    ans_en = proc.decode(output[0], skip_special_tokens=True).strip()
    ans_ar = smart_translate_answer(ans_en)

    return q_ar, q_en, ans_ar, ans_en


# ---------------------------------------------------------
# 4) Streamlit UI
# ---------------------------------------------------------
st.set_page_config(
    page_title="نموذج ثنائي اللغة للأشعة (CT/X-Ray/MRI)",
    layout="wide",
)

st.title("🩻 نموذج ثنائي اللغة (عربي – إنجليزي) لأسئلة صور الأشعة")
st.markdown("ارفع صورة طبية واسأل بالعربية أو الإنجليزية، "
            "وستحصل على الإجابة باللغتين.")

col_left, col_right = st.columns([1, 2])

with col_left:
    uploaded = st.file_uploader("🔍 ارفع صورة الأشعة", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="الصورة المرفوعة", use_column_width=True)
    else:
        img = None

with col_right:
    question_txt = st.text_input("💬 أدخل سؤالك (بالعربية أو الإنجليزية)")

    if st.button("إرسال"):
        q_ar, q_en, a_ar, a_en = vqa_multilingual(img, question_txt)

        st.text_area("🟠 السؤال بالعربية", q_ar, height=60)
        st.text_area("🟢 السؤال بالإنجليزية", q_en, height=60)
        st.text_area("🟠 الإجابة بالعربية", a_ar, height=60)
        st.text_area("🟢 الإجابة بالإنجليزية", a_en, height=60)
