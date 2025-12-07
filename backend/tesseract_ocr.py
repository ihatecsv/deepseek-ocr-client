#!/usr/bin/env python3
"""
Tesseract OCR Module
CPU-based OCR alternative to DeepSeek-OCR
"""
import os
import logging
import shutil
from PIL import Image

logger = logging.getLogger(__name__)

# Check if pytesseract is available
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. Install with: pip install pytesseract")


def check_tesseract_availability():
    """Check if Tesseract OCR is installed and available"""
    if not TESSERACT_AVAILABLE:
        return {
            'available': False,
            'error': 'pytesseract Python package not installed'
        }
    
    # Check if tesseract executable is available
    tesseract_cmd = shutil.which('tesseract')
    if tesseract_cmd is None:
        # Try common Windows installation paths
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                tesseract_cmd = path
                break
    
    if tesseract_cmd is None:
        return {
            'available': False,
            'error': 'Tesseract executable not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki'
        }
    
    try:
        # Try to get version to verify it works
        version = pytesseract.get_tesseract_version()
        return {
            'available': True,
            'version': str(version),
            'path': tesseract_cmd
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }


def perform_tesseract_ocr(image_path, lang='eng+ara', output_dir=None):
    """
    Perform OCR on an image using Tesseract
    
    Args:
        image_path: Path to the image file
        lang: Language(s) for OCR (default: 'eng+ara' for English and Arabic)
        output_dir: Optional directory to save results
    
    Returns:
        dict with 'status', 'result', and optionally 'error'
    """
    if not TESSERACT_AVAILABLE:
        return {
            'status': 'error',
            'error': 'pytesseract not installed'
        }
    
    availability = check_tesseract_availability()
    if not availability['available']:
        return {
            'status': 'error',
            'error': availability['error']
        }
    
    try:
        # Open image
        image = Image.open(image_path)
        
        # Perform OCR
        logger.info(f"Running Tesseract OCR on {image_path} with lang={lang}")
        
        # Get text with detailed data
        text = pytesseract.image_to_string(image, lang=lang)
        
        # Save result if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            result_path = os.path.join(output_dir, 'result.txt')
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved OCR result to {result_path}")
        
        return {
            'status': 'success',
            'result': text,
            'prompt_type': 'tesseract',
            'raw_tokens': None,
            'boxes_image_path': None
        }
        
    except Exception as e:
        logger.error(f"Tesseract OCR error: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def get_available_languages():
    """Get list of available Tesseract languages"""
    if not TESSERACT_AVAILABLE:
        return []
    
    availability = check_tesseract_availability()
    if not availability['available']:
        return []
    
    try:
        return pytesseract.get_languages()
    except Exception:
        return ['eng']  # Default fallback
