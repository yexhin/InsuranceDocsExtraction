import json
import cv2
import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from .preprocessor import preprocess_for_ocr
import pytesseract
import google.generativeai as genai


class InsuranceDocExtractor:
    def __init__(self, gemini_api_key: str):
        self.api = gemini_api_key
        # Using model gemini-2.0-flash
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.logger = logging.getLogger(__name__)
       
    
    def pdf_to_images(self, path, dpi=300) -> List[Image.Image]:
        try:
            images = convert_from_path(path, dpi=dpi,
                                       poppler_path=r"D:\Release-24.07.0-0\poppler-24.07.0\Library\bin")
            self.logger.info(f"Converted PDF to {len(images)} images")
            return images
        except Exception as e:
            self.logger.error(f"Error converting PDF: {str(e)}")
            raise
            
    def ocr_chinese(self, images: List[Image.Image]) -> str:
        full_text = ""
        for i, image in enumerate(images):
            try:
                # Convert PIL to numpy for preprocessing
                img_array = np.array(image)
                # Nếu preprocess_for_ocr nhận numpy array, truyền trực tiếp
                preprocessed_image = preprocess_for_ocr(img_array)
                pil_image = Image.fromarray(preprocessed_image)
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                page_text = pytesseract.image_to_string(
                    pil_image,
                    lang='chi_sim+chi_tra+eng',
                    config=custom_config
                )
                full_text += f"\n=== Page {i+1} ===\n{page_text}\n"
                self.logger.info(f"OCR completed for page {i+1}")
            except Exception as e:
                self.logger.error(f"OCR error on page {i+1}: {str(e)}")
                continue
        return full_text.strip()
        
    
    
    def extractor_process(self, images: List[Image.Image], ocr_text: str) -> Dict[str, Any]:
        system_instruction = """
            You are an expert Insurance Documents processor who can extract key information accurately.

            Your task is Extract structured information from insurance documents using both visual analysis and OCR text.

            Return ONLY valid JSON with these fields, break into 2 parts:
            {
                Part 1: PART I-ТО ВE СОМPLETED BY THE INSURED / POLICYOWNER:
            {
                "name_of_policyowner": "保單持有人姓名",
                "name_of_insured": "受保人姓名",
                "HKID": "香港身份證號碼",
                "date_of_birth": "出生日期 (YYYY-MM-DD)",
                "sex": "性別",
                "Did you submit this insurance claims to other insurance company?": "你是否向其他保險公司提出此索償？ (Yes/No)",
                "Have you had any prior treatment for this related condition?": "你是否曾就有關情況接受任何治療？ (Yes/No)",
                "Was the hospitalization / surgery a result of an accident?": "是"
            },
            {
                "payment_information": {
                    "name_of_account_holder": "賬戶持有人姓名",
                    "bank_name": "銀行名稱",
                    "bank_number": "銀行賬號",
            }
            }
            
            
            { 
            PART II — TO BE COMPLETED BY THE ATTENDING PHYSICIAN / SURGEON AT THE CLAIMANT'S OWN EXPENSES
            "patient_info": {
                "patient_name_in_full": "病人姓名",
            },
            "admission_info": {
                "date_of_admission": "入院日期 (YYYY-MM-DD)",
                "date_of_discharge": "出院日期 (YYYY-MM-DD)",
                "period_in_icu": "入住深切治療部日期",
                "level_of_hospital_ward": "病房級別",
            },
            "clinical_history": {
                "first_consultation_date": "首次求診日期 (YYYY-MM-DD)",
                "symptom_duration": "症狀持續時間",
                "symptoms": "主訴 / 症狀"
            },
            "hospitalization_details": {
                "hospital_name": "醫院名稱",
                "final_diagnosis": "最終診斷",
                "date_of_operation": "手術日期 (YYYY-MM-DD)",
                "operation_procedures": "手術名稱"
            }
            }

            PROCESSING RULES:
            1. Use both image analysis and OCR text fully to enhance the accuracy.
            2. Extract both chinese and english characters carefully.
            3. Convert dates to YYYY-MM-DD.
            4. Return only valid JSON.
            """
        try:
            # Use main page (usually most important) + OCR text
            main_image = images[0]
            
            # Create multimodal content
            content_parts = [
                system_instruction,
                main_image,
                f"\n\nOCR EXTRACTED TEXT:\n{ocr_text}"
            ]
            
            
            if len(images) > 1:
                for i, img in enumerate(images[1:], 2):  
                    content_parts.extend([
                        f"\n--- Additional Page {i} ---",
                        img
                    ])
            
            # Generate with Gemini 2.0 Flash
            response = self.model.generate_content(
                content_parts,
                generation_config={
                    "temperature": 0.1,  # Low temperature for accuracy
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 4096,
                }
            )
            
            # Clean and parse response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            # Parse JSON
            extracted_data = json.loads(response_text)
            
            self.logger.info("Successfully extracted data with Gemini 2.0 Flash")
            return extracted_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            return {
                "error": "Failed to parse extracted data",
                "raw_response": response_text,
                "error_type": "json_decode_error"
            }
        
        except Exception as e:
            self.logger.error(f"Gemini 2.0 Flash API error: {str(e)}")
            return {
                "error": f"API error: {str(e)}",
                "error_type": "api_error"
            }

    def process_document(self, pdf_path: str, dpi: int = 300, use_ocr: bool = True) -> Dict[str, Any]:
        """Complete pipeline optimized for Gemini 2.0 Flash"""
        try:
            self.logger.info(f"Processing document: {pdf_path}")
            # Step 1: Convert PDF to images
            self.logger.info("Converting PDF to images...")
            images = self.pdf_to_images(pdf_path, dpi)
            # Step 2: OCR processing 
            ocr_text = ""
            if use_ocr:
                self.logger.info("Performing OCR...")
                ocr_text = self.ocr_chinese(images)
            # Step 3: Gemini 2.0 Flash multimodal extraction
            self.logger.info("Extracting with Gemini 2.0 Flash")
            extracted_data = self.extractor_process(images, ocr_text)
            # Step 4: Quality assessment
            quality_score = None
            if hasattr(self, "assess_extraction_quality"):
                quality_score = self.assess_extraction_quality(extracted_data)
            # Compile results
            result = {
                "status": "success",
                "model_used": "gemini-2.0-flash",
                "extracted_data": extracted_data,
                "ocr_text": ocr_text if use_ocr else None,
                "quality_assessment": quality_score,
                "metadata": {
                    "pages_processed": len(images),
                    "dpi_used": dpi,
                    "ocr_enabled": use_ocr,
                    "text_length": len(ocr_text) if ocr_text else 0,
                    "chinese_chars": len([c for c in ocr_text if '\u4e00' <= c <= '\u9fff']) if ocr_text else 0
                }
            }
            self.logger.info("Document processing completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "model_used": "gemini-2.0-flash-exp",
                "extracted_data": None
            }

