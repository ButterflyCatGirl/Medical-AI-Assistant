# Medical AI Assistant

A Streamlit web application that provides medical image analysis and Arabic-to-English translation using fine-tuned AI models.

## Features

- 🔍 **Medical Image Analysis**: Upload medical images and ask questions about findings
- 🌐 **Arabic Translation**: Translate medical text from Arabic to English
- 🤖 **AI-Powered**: Uses fine-tuned medical VQA models

## Deployment

This app is designed for deployment on Streamlit Cloud:

1. Create a new GitHub repository
2. Add these files to the repository
3. Connect to Streamlit Cloud
4. Deploy directly from GitHub

## Models Used

- Medical VQA: `Mohamed264/llava-medical-VQA-lora-merged3`
- Translation: `Helsinki-NLP/opus-mt-ar-en`

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
