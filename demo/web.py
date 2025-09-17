import streamlit as st
import os
import json
from io import BytesIO
from components.insuranceExtractor import InsuranceDocExtractor

st.set_page_config(page_title="Medical Insurance Claim Extractor", layout="wide")
st.title("Medical Insurance Claim Extractor")

# Lấy API key từ secrets
api_key = st.secrets["google"]["GOOGLE_API_KEY"]

# Tạo file credential tạm thời từ secret
cred_json = json.loads(st.secrets["google"]["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
cred_path = "temp_credentials.json"
with open(cred_path, "w", encoding="utf-8") as f:
    json.dump(cred_json, f)

# Trỏ biến môi trường
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

if not api_key:
    st.error("GEMINI_API_KEY not found in environment variables.")
    st.stop()

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is None:
    st.info("Upload a PDF file to start extraction.")
    st.stop()

# Save temp PDF
temp_path = "temp_uploaded.pdf"
with open(temp_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

# Initialize extractor
extractor = InsuranceDocExtractor(gemini_api_key=api_key)

# Convert PDF → images
with st.spinner("Converting PDF to images..."):
    images = extractor.pdf_to_images(temp_path, dpi=300)
    st.success(f"PDF converted to {len(images)} pages")

# OCR
with st.spinner("Performing OCR..."):
    ocr_text = extractor.ocr_chinese(images)
    st.success("OCR completed")

# Extract structured data
with st.spinner("Extracting structured data with Gemini..."):
    extracted_data = extractor.process_document(temp_path, dpi=300, use_ocr=True)["extracted_data"]
    st.success("Extraction completed")

with st.expander("Extracted JSON"):
    st.json(extracted_data)

# Download options
json_bytes = BytesIO(json.dumps(extracted_data, indent=2, ensure_ascii=False).encode("utf-8"))
st.download_button("Download Extracted JSON", data=json_bytes, file_name="extracted_data.json", mime="application/json")

# Cleanup
os.remove(temp_path)
