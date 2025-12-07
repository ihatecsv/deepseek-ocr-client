#!/usr/bin/env python3
"""
TTS Backend Module
Supports Edge TTS (CPU) and Coqui XTTS (GPU)
"""
import os
import logging
import tempfile
import asyncio

logger = logging.getLogger(__name__)

# Check for Coqui TTS availability (GPU)
COQUI_AVAILABLE = False
try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except ImportError:
    logger.info("Coqui TTS not available. GPU TTS will be disabled.")

# Edge TTS is always available (CPU, uses API)
EDGE_TTS_AVAILABLE = False
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    logger.warning("edge-tts not installed. Run: pip install edge-tts")


# Language to Edge TTS voice mapping
EDGE_VOICES = {
    'ar': 'ar-SA-HamedNeural',      # Arabic (Saudi Arabia)
    'ar-EG': 'ar-EG-SalmaNeural',   # Arabic (Egypt)
    'en': 'en-US-JennyNeural',      # English (US)
    'en-GB': 'en-GB-SoniaNeural',   # English (UK)
    'fr': 'fr-FR-DeniseNeural',     # French
    'de': 'de-DE-KatjaNeural',      # German
    'es': 'es-ES-ElviraNeural',     # Spanish
    'zh': 'zh-CN-XiaoxiaoNeural',   # Chinese
    'ja': 'ja-JP-NanamiNeural',     # Japanese
    'ko': 'ko-KR-SunHiNeural',      # Korean
    'ru': 'ru-RU-SvetlanaNeural',   # Russian
    'default': 'en-US-JennyNeural'
}


def detect_language(text: str) -> str:
    """
    Simple language detection based on character ranges.
    Returns language code.
    """
    if not text:
        return 'en'
    
    # Check for Arabic characters
    arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    # Check for Chinese characters
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    # Check for Japanese (Hiragana/Katakana)
    japanese_count = sum(1 for c in text if '\u3040' <= c <= '\u30ff')
    # Check for Korean
    korean_count = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
    # Check for Cyrillic (Russian)
    cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    
    total = len(text)
    threshold = 0.1  # 10% of text
    
    if arabic_count / total > threshold:
        return 'ar'
    elif chinese_count / total > threshold:
        return 'zh'
    elif japanese_count / total > threshold:
        return 'ja'
    elif korean_count / total > threshold:
        return 'ko'
    elif cyrillic_count / total > threshold:
        return 'ru'
    else:
        return 'en'


async def edge_tts_generate(text: str, output_path: str, language: str = None) -> dict:
    """
    Generate speech using Edge TTS (CPU, API-based).
    
    Args:
        text: Text to convert to speech
        output_path: Path to save the audio file
        language: Language code (auto-detected if None)
    
    Returns:
        dict with status, path, and info
    """
    if not EDGE_TTS_AVAILABLE:
        return {
            'status': 'error',
            'message': 'edge-tts not installed. Run: pip install edge-tts'
        }
    
    try:
        # Auto-detect language if not specified
        if not language:
            language = detect_language(text)
        
        # Get voice for language
        voice = EDGE_VOICES.get(language, EDGE_VOICES['default'])
        
        logger.info(f"Edge TTS: Using voice {voice} for language {language}")
        
        # Generate audio
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        
        return {
            'status': 'success',
            'path': output_path,
            'voice': voice,
            'language': language,
            'engine': 'edge_tts'
        }
    except Exception as e:
        logger.error(f"Edge TTS error: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }


def coqui_tts_generate(text: str, output_path: str, language: str = None) -> dict:
    """
    Generate speech using Coqui XTTS (GPU).
    
    Args:
        text: Text to convert to speech
        output_path: Path to save the audio file
        language: Language code
    
    Returns:
        dict with status, path, and info
    """
    if not COQUI_AVAILABLE:
        return {
            'status': 'error',
            'message': 'Coqui TTS not installed. Run: pip install TTS'
        }
    
    try:
        # Auto-detect language if not specified
        if not language:
            language = detect_language(text)
        
        # Map to Coqui language codes
        coqui_lang_map = {
            'ar': 'ar',
            'en': 'en',
            'fr': 'fr',
            'de': 'de',
            'es': 'es',
            'zh': 'zh-cn',
            'ja': 'ja',
            'ko': 'ko',
            'ru': 'ru'
        }
        coqui_lang = coqui_lang_map.get(language, 'en')
        
        logger.info(f"Coqui TTS: Using language {coqui_lang}")
        
        # Initialize XTTS model
        tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Generate audio
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            language=coqui_lang
        )
        
        return {
            'status': 'success',
            'path': output_path,
            'language': coqui_lang,
            'engine': 'coqui_xtts'
        }
    except Exception as e:
        logger.error(f"Coqui TTS error: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }


def check_tts_availability() -> dict:
    """
    Check which TTS engines are available.
    """
    return {
        'edge_tts': EDGE_TTS_AVAILABLE,
        'coqui_xtts': COQUI_AVAILABLE
    }
