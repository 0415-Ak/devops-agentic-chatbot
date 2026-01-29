# ocr_tool.py

import os
import json
import cv2
import pytesseract
import pandas as pd
from docx import Document
import pdfplumber
from pdf2image import convert_from_path
import numpy as np


# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# POPPLER_PATH = r"C:\Users\aksha\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"


def extract_text_from_pdf(path):
    text = ""
    try:
        # Convert PDF to images (each page = one image)
        pages = convert_from_path(path, dpi=300)

        for i, page in enumerate(pages, start=1):
            # Convert PIL image to OpenCV format
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

            # Preprocess for better OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            gray = cv2.medianBlur(gray, 3)

            # OCR
            custom_config = r'--oem 3 --psm 6'
            page_text = pytesseract.image_to_string(gray, config=custom_config)

            text += f"\n--- Page {i} ---\n{page_text}\n"

        return text.strip()

    except Exception as e:
        return f"⚠️ Error extracting text with OCR: {str(e)}"



def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_excel(path):
    dfs = pd.read_excel(path, sheet_name=None)
    text = ""
    for sheet, df in dfs.items():
        text += f"\n--- Sheet: {sheet} ---\n"
        text += df.to_string(index=False)
    return text

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray, config=custom_config)
    return text



def extract_text_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return json.dumps(data, indent=2)

def extract_text_from_file(path):
    ext = os.path.splitext(path)[-1].lower()
    try:
        if ext in [".png", ".jpg", ".jpeg"]:
            return extract_text_from_image(path)
        elif ext == ".pdf":
            return extract_text_from_pdf(path)
        elif ext == ".docx":
            return extract_text_from_docx(path)
        elif ext == ".json":
            return extract_text_from_json(path)
        elif ext in [".txt", ".py", ".js", ".html", ".md"]:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == ".xlsx":
            return extract_text_from_excel(path)
        else:
            return f"❌ Unsupported file type: {ext}"
    except Exception as e:
        return f"⚠️ Error extracting text: {str(e)}"


print("done")