import cv2
import numpy as np

def enhance_img(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Improve contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Adjust the config
    alpha = 1.2  # contrast
    beta = 10    # brightness
    adjusted = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    return adjusted

def reduce_noise(img):
    # remove noise
    denoised = cv2.GaussianBlur(img, (3,3), 0)
    
    # preserve edges
    denoised = cv2.bilateralFilter(img, 9, 75, 75)
    
    return denoised

def deskew_img(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), 
                           flags=cv2.INTER_CUBIC, 
                           borderMode=cv2.BORDER_REPLICATE)
    return rotated

def binarize_img(img):
    # Otsu thresholding
    _, binary = cv2.threshold(img, 0, 255, 
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def preprocess_for_ocr(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Enhance contrast
    enhanced = enhance_img(image)
    
    # Reduce noise
    denoised = reduce_noise(enhanced)
    
    # Deskew
    deskewed = deskew_img(denoised)
    
    # Binarize
    binary = binarize_img(deskewed)
    
    # Optional: Morphological operations
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned
