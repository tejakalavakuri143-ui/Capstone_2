import os
import json
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance
import pdf2image
import docx
import re

# Set the Tesseract path for Windows if necessary
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def clean_txt(text):
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _ocr_image_obj(img):
    """
    Run Tesseract on a PIL Image with pre-processing for better accuracy.
    Upscales to 2x and sharpens before OCR — critical for scanned/handwritten docs.
    """
    img_up    = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    img_sharp = ImageEnhance.Sharpness(img_up).enhance(2.0)
    config    = "--oem 3 --psm 6"
    return pytesseract.image_to_string(img_sharp, config=config)


def extract_from_pdf(file_path):
    """
    Extract text from a PDF.

    Strategy:
      1. Try pdfplumber (fast, accurate for text-layer PDFs).
      2. If pdfplumber returns nothing — scanned / image-only PDF —
         rasterise each page with pdf2image and run Tesseract OCR.
    """
    # --- Pass 1: pdfplumber (text layer) ---
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[Extractor] pdfplumber warning on {file_path}: {e}")

    if text.strip():
        return text

    # --- Pass 2: OCR fallback (scanned / image-only PDF) ---
    print(f"[Extractor] No text layer found — falling back to OCR: {os.path.basename(file_path)}")
    try:
        pages = pdf2image.convert_from_path(file_path, dpi=200)
        ocr_text = ""
        for page_img in pages:
            ocr_text += _ocr_image_obj(page_img) + "\n"
        return ocr_text
    except Exception as e:
        print(f"[Extractor] OCR fallback failed for {file_path}: {e}")
        return ""


def extract_from_docx(file_path):
    doc  = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


def extract_from_image(file_path):
    """
    Extract text from an image file (PNG / JPG / JPEG / WEBP).
    Uses upscaling + sharpening for better OCR on low-res scans.
    """
    img = Image.open(file_path)
    return _ocr_image_obj(img)


def detect_file_type(file_path):
    """
    Detect file type by extension.
    NOTE: .jpeg must include the dot to avoid false matches.
    """
    fp = file_path.lower()
    if fp.endswith(".pdf"):
        return "pdf"
    elif fp.endswith(".docx"):
        return "docx"
    elif fp.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return "image"
    else:
        return "unsupported"
   
def extractor_agent(files):
    #input_folder = "data/incoming"
    results = []

    #if not os.path.exists(input_folder):
     #   return {"errors" : f"Folder not found {input_folder}"}
    
   # files = os.listdir(input_folder)
    #for file_name in files:
    for file_path in files:
        #file_path = os.path.join(input_folder, file_name)
        file_name = os.path.basename(file_path)
        
        if not os.path.isfile(file_path):
            continue
        
        file_type = detect_file_type(file_path)
        try:
            if file_type == "pdf":
                text = extract_from_pdf(file_path)
            elif file_type == "docx":
                text = extract_from_docx(file_path)
            elif file_type == "image":
                text = extract_from_image(file_path)
            else:
                print(f"Unsupported file type: {file_name}")
                continue
            
            if not text or text.strip() == "":
                print(f"[WARNING] no text extracted")
                continue

            cleaned_text = clean_txt(text)
            results.append({
                "file_name": file_name,
                "file_type": file_type,
                "language" : "unknown",
                "file_text": cleaned_text
            })

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue  # Continue processing other files

    return {"extracted_data": results}
"""
if __name__ == "__main__":
    output = extractor_agent()

    print("OUTPUT IS:")
    print(json.dumps(output, indent=4, ensure_ascii=False))"""





