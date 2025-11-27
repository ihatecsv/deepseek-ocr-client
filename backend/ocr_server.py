#!/usr/bin/env python3
"""
DeepSeek OCR Backend Server
Handles model loading, caching, and OCR inference
"""
import os
import sys
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoModel, AutoTokenizer
import tempfile
import time
from threading import Thread, Lock
import fitz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from werkzeug
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'

# Use local cache directory relative to the app
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, '..', 'cache')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models')
OUTPUT_DIR = os.path.join(CACHE_DIR, 'outputs')

# Device preference: 'auto' | 'cpu' | 'gpu'
device_preference = 'auto'

# Progress tracking
progress_data = {
    'status': 'idle',  # idle, loading, loaded, error
    'stage': '',       # tokenizer, model
    'message': '',
    'progress_percent': 0,  # 0-100
    'chars_generated': 0,  # For OCR character counting
    'raw_token_stream': '',  # Accumulated raw tokens during OCR
    'timestamp': time.time()
}
progress_lock = Lock()
loading_thread = None

def update_progress(status, stage='', message='', progress_percent=0, chars_generated=0, raw_token_stream=''):
    """Update the global progress data"""
    global progress_data
    with progress_lock:
        progress_data['status'] = status
        progress_data['stage'] = stage
        progress_data['message'] = message
        progress_data['progress_percent'] = progress_percent
        progress_data['chars_generated'] = chars_generated
        progress_data['raw_token_stream'] = raw_token_stream
        progress_data['timestamp'] = time.time()
        if chars_generated > 0:
            logger.info(f"Progress: {status} - {stage} - {message} ({progress_percent}%) - {chars_generated} chars")
        else:
            logger.info(f"Progress: {status} - {stage} - {message} ({progress_percent}%)")

def check_gpu_availability():
    """Check if CUDA is available"""
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        logger.warning("No GPU available, will use CPU (this will be slow!)")
        return False

def get_cache_dir_size(directory):
    """Get total size of files in directory in bytes"""
    total = 0
    try:
        for entry in os.scandir(directory):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_cache_dir_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total

def load_model_background():
    """Background thread function to load the model"""
    global model, tokenizer

    try:
        update_progress('loading', 'init', 'Initializing model loading...', 0)
        logger.info(f"Loading DeepSeek OCR model from {MODEL_NAME}...")
        logger.info(f"Model will be cached in: {MODEL_CACHE_DIR}")

        # Create cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        # Check GPU availability
        has_gpu = check_gpu_availability()

        # Load tokenizer (10% progress)
        update_progress('loading', 'tokenizer', 'Loading tokenizer...', 10)
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR
        )
        update_progress('loading', 'tokenizer', 'Tokenizer loaded', 20)

        # Check if model is already cached
        initial_cache_size = get_cache_dir_size(MODEL_CACHE_DIR)
        is_cached = initial_cache_size > 100 * 1024 * 1024  # More than 100 MB suggests model is cached

        if is_cached:
            # Model is cached, just loading from disk
            update_progress('loading', 'model', 'Loading model from cache...', 25)
            logger.info("Loading model from cache...")
        else:
            # Model needs to be downloaded
            update_progress('loading', 'model', 'Downloading model files (this will take several minutes)...', 25)
            logger.info("Downloading model (this may take a while on first run)...")

        # Start a thread to monitor download progress (only if downloading)
        download_monitor_active = [True]  # Use list for mutable access in nested function
        def monitor_download():
            last_size = initial_cache_size
            stall_count = 0
            progress = 25

            while download_monitor_active[0] and progress < 75:
                time.sleep(2)  # Check every 2 seconds
                current_size = get_cache_dir_size(MODEL_CACHE_DIR)

                if current_size > last_size:
                    # Download is progressing
                    stall_count = 0
                    # Increment progress (max 75%)
                    progress = min(progress + 2, 75)
                    size_mb = current_size / (1024 * 1024)
                    update_progress('loading', 'model', f'Downloading model files... ({size_mb:.1f} MB downloaded)', progress)
                    last_size = current_size
                else:
                    # No change in size
                    stall_count += 1
                    if stall_count < 5:  # Still show activity for first 10 seconds
                        if is_cached:
                            update_progress('loading', 'model', 'Loading model from cache...', progress)
                        else:
                            update_progress('loading', 'model', 'Downloading model files...', progress)

        monitor_thread = Thread(target=monitor_download)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Try to use flash attention if available, otherwise fallback
        try:
            # 首先检查transformers版本是否支持flash_attention_2
            import transformers
            transformers_version = transformers.__version__
            logger.info(f"Transformers version: {transformers_version}")

            try:
                import transformers.models.llama.modeling_llama as modeling_llama
                if not hasattr(modeling_llama, 'LlamaFlashAttention2'):
                    class LlamaFlashAttention2:
                        pass
                    modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2
            except Exception as _e:
                logger.warning(f"Flash attention shim failed: {_e}")

            # 尝试使用flash attention 2
            try:
                model = AutoModel.from_pretrained(
                    MODEL_NAME,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True,
                    cache_dir=MODEL_CACHE_DIR
                )
                logger.info("Using flash attention 2")
            except Exception as e:
                logger.warning(f"Flash attention 2 not supported with current transformers version: {e}")
                logger.info("Falling back to default attention implementation")
                model = AutoModel.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True,
                    use_safetensors=True,
                    cache_dir=MODEL_CACHE_DIR
                )
        except Exception as e:
            logger.warning(f"Error loading model: {e}")
            raise e

        # Stop download monitor and wait for it to finish
        download_monitor_active[0] = False
        monitor_thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish

        # Set to eval mode (80% progress)
        update_progress('loading', 'gpu', 'Moving model to GPU...', 80)
        model = model.eval()

        # Move to GPU if available (90% progress)
        update_progress('loading', 'gpu', 'Optimizing model...', 90)
        if device_preference == 'cpu':
            logger.info("Device preference set to CPU")
        elif device_preference == 'gpu' and not has_gpu:
            logger.warning("GPU requested but not available; falling back to CPU")
        if has_gpu and device_preference != 'cpu':
            model = model.cuda().to(torch.bfloat16)
            logger.info("Model loaded on GPU with bfloat16")
        else:
            logger.info("Model loaded on CPU (inference will be slower)")

        logger.info("Model loaded successfully!")
        update_progress('loaded', 'complete', 'Model ready!', 100)

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        update_progress('error', 'failed', str(e), 0)
        import traceback
        traceback.print_exc()
        # 确保在错误情况下重置模型和分词器
        model = None
        tokenizer = None

def load_model():
    """Load the DeepSeek OCR model and tokenizer

    Model size configurations:
    - Tiny: base_size=512, image_size=512, crop_mode=False
    - Small: base_size=640, image_size=640, crop_mode=False
    - Base: base_size=1024, image_size=1024, crop_mode=False
    - Large: base_size=1280, image_size=1280, crop_mode=False
    - Gundam (recommended): base_size=1024, image_size=640, crop_mode=True
    """
    global model, tokenizer, loading_thread

    if model is not None and tokenizer is not None:
        logger.info("Model already loaded")
        update_progress('loaded', 'complete', 'Model already loaded', 100)
        return True

    # Check if already loading
    if loading_thread is not None and loading_thread.is_alive():
        logger.info("Model loading already in progress")
        # 等待加载完成
        loading_thread.join(timeout=300)  # 等待最多5分钟
        if model is not None and tokenizer is not None:
            return True
        else:
            logger.error("Model loading timed out or failed")
            return False

    # Start loading in background thread
    loading_thread = Thread(target=load_model_background)
    loading_thread.daemon = True
    loading_thread.start()

    # 等待加载完成
    loading_thread.join(timeout=300)  # 等待最多5分钟
    if model is not None and tokenizer is not None:
        return True
    else:
        logger.error("Model loading failed")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'gpu_available': torch.cuda.is_available()
    })

@app.route('/progress', methods=['GET'])
def get_progress():
    """Get current model loading progress"""
    with progress_lock:
        current_data = progress_data.copy()
    return jsonify(current_data)

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Endpoint to trigger model loading"""
    global device_preference
    try:
        data = request.get_json(silent=True) or {}
        if 'force_cpu' in data and data['force_cpu']:
            device_preference = 'cpu'
        else:
            device_preference = 'auto'
    except Exception:
        device_preference = 'auto'

    success = load_model()
    if success:
        return jsonify({'status': 'success', 'message': 'Model loaded successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to load model'}), 500

@app.route('/ocr', methods=['POST'])
def perform_ocr():
    """Perform OCR on uploaded image"""
    global model, tokenizer

    try:
        # Check if model is loaded
        if model is None or tokenizer is None:
            logger.info("Model not loaded, loading now...")
            if not load_model():
                logger.error("Failed to load model")
                return jsonify({'status': 'error', 'message': 'Failed to load model. Please check logs for details.'}), 500

        # 再次检查模型是否已成功加载
        if model is None or tokenizer is None:
            logger.error("Model is still None after loading attempt")
            return jsonify({'status': 'error', 'message': 'Model loading failed. Please restart the application.'}), 500

        # Get image from request
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image provided'}), 400

        image_file = request.files['image']

        # Get optional parameters
        prompt_type = request.form.get('prompt_type', 'document')
        base_size = int(request.form.get('base_size', 1024))
        image_size = int(request.form.get('image_size', 640))
        crop_mode = request.form.get('crop_mode', 'true').lower() == 'true'

        # Define prompts and their expected output file extensions
        prompt_configs = {
            'document': {
                'prompt': '<image>\n<|grounding|>Convert the document to markdown. ',
                'output_file': 'result.mmd'
            },
            'ocr': {
                'prompt': '<image>\n<|grounding|>OCR this image. ',
                'output_file': 'result.txt'
            },
            'free': {
                'prompt': '<image>\nFree OCR. ',
                'output_file': 'result.txt'
            },
            'figure': {
                'prompt': '<image>\nParse the figure. ',
                'output_file': 'result.txt'
            },
            'describe': {
                'prompt': '<image>\nDescribe this image in detail. ',
                'output_file': 'result.txt'
            }
        }

        config = prompt_configs.get(prompt_type, prompt_configs['document'])
        prompt = config['prompt']
        expected_output_file = config['output_file']

        logger.info(f"Processing OCR request with prompt type: {prompt_type}")

        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image_file.save(tmp_file.name)
            temp_image_path = tmp_file.name

        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Perform inference - save results to files with token counting
        logger.info("Running OCR inference...")
        logger.info(f"Saving results to: {OUTPUT_DIR}")

        # Reset progress
        update_progress('processing', 'ocr', 'Starting OCR...', 0, 0)

        # Capture stdout to count characters and collect raw tokens
        old_stdout = sys.stdout
        char_count = [0]  # Use list for mutable access in nested function

        class CharCountingStream:
            def __init__(self, original_stdout):
                self.original = original_stdout
                self.accumulated_text = ''  # Accumulate all text
                self.section_count = 0  # Count === markers to track sections

            def write(self, text):
                # Handle Unicode encoding errors gracefully
                try:
                    self.original.write(text)
                except UnicodeEncodeError:
                    # If stdout can't handle the character, encode with error replacement
                    try:
                        safe_text = text.encode(self.original.encoding or 'utf-8', errors='replace').decode(self.original.encoding or 'utf-8')
                        self.original.write(safe_text)
                    except:
                        # If all else fails, just skip writing to original stdout
                        pass

                # Accumulate text exactly as received (preserves all formatting)
                self.accumulated_text += text

                # Count === markers to determine sections
                # Section 0-1: Before first ===
                # Section 1-2: BASE/PATCHES info
                # Section 2-3: Token generation (what we want)
                # Section 3+: Compression stats
                self.section_count = self.accumulated_text.count('=' * 20)  # Count long === lines

                # Extract and process token section (between 2nd and 3rd ===)
                if self.section_count >= 2:
                    # Find the token section
                    parts = self.accumulated_text.split('=' * 20)
                    if len(parts) >= 3:
                        # Token section is between 2nd and 3rd === markers
                        token_section = parts[2]

                        # Store the raw token section, removing any leading/trailing = and whitespace
                        raw_token_text = token_section.strip().lstrip('=').strip()

                        # Count characters in the raw token text
                        char_count[0] = len(raw_token_text)

                        # Update progress with the raw token stream (no artificial newlines)
                        if char_count[0] > 0:
                            update_progress('processing', 'ocr', 'Generating OCR...', 50, char_count[0], raw_token_text)

            def flush(self):
                self.original.flush()

        char_stream = CharCountingStream(old_stdout)
        sys.stdout = char_stream

        try:
            model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=OUTPUT_DIR,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=True,
                test_compress=True
            )
        finally:
            sys.stdout = old_stdout
            update_progress('idle', '', '', 0, 0)  # Reset progress

        logger.info("OCR inference completed successfully")

        # Read the expected output file based on prompt type
        result_filepath = os.path.join(OUTPUT_DIR, expected_output_file)
        result_text = None

        logger.info(f"Looking for output file: {expected_output_file}")
        logger.info(f"Files in output dir: {os.listdir(OUTPUT_DIR)}")

        if os.path.exists(result_filepath):
            with open(result_filepath, 'r', encoding='utf-8') as f:
                result_text = f.read()
            logger.info(f"Successfully read result from: {expected_output_file}")
            logger.info(f"Result text (first 200 chars): {result_text[:200]}")
        else:
            # Fallback: try to find any text-like file
            logger.warning(f"Expected file '{expected_output_file}' not found, searching for alternatives")
            for filename in os.listdir(OUTPUT_DIR):
                if filename.endswith(('.txt', '.mmd', '.md')):
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result_text = f.read()
                    logger.info(f"Read result from alternative file: {filename}")
                    break

        if result_text is None:
            result_text = "OCR completed but no text file was generated"
            logger.warning("No result file found in output directory")

        # Check for boxes image (result_with_boxes.jpg)
        boxes_image_path = os.path.join(OUTPUT_DIR, 'result_with_boxes.jpg')
        has_boxes_image = os.path.exists(boxes_image_path)

        logger.info(f"Boxes image exists: {has_boxes_image}")

        # Clean up temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Extract raw token text from the stream (between 2nd and 3rd === markers)
        raw_token_text = None
        if char_stream.section_count >= 2:
            parts = char_stream.accumulated_text.split('=' * 20)
            if len(parts) >= 3:
                raw_token_text = parts[2].strip().lstrip('=').strip()

        return jsonify({
            'status': 'success',
            'result': result_text,
            'boxes_image_path': 'result_with_boxes.jpg' if has_boxes_image else None,
            'prompt_type': prompt_type,
            'raw_tokens': raw_token_text
        })

    except Exception as e:
        logger.error(f"Error during OCR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/ocr_pdf', methods=['POST'])
def perform_ocr_pdf():
    """Perform OCR on uploaded PDF (per-page)"""
    global model, tokenizer

    try:
        if model is None or tokenizer is None:
            logger.info("Model not loaded, loading now...")
            if not load_model():
                logger.error("Failed to load model")
                return jsonify({'status': 'error', 'message': 'Failed to load model. Please check logs for details.'}), 500

        # 再次检查模型是否已成功加载
        if model is None or tokenizer is None:
            logger.error("Model is still None after loading attempt")
            return jsonify({'status': 'error', 'message': 'Model loading failed. Please restart the application.'}), 500

        if 'pdf' not in request.files:
            return jsonify({'status': 'error', 'message': 'No PDF provided'}), 400

        pdf_file = request.files['pdf']
        prompt_type = request.form.get('prompt_type', 'document')
        base_size = int(request.form.get('base_size', 1024))
        image_size = int(request.form.get('image_size', 640))
        crop_mode = request.form.get('crop_mode', 'true').lower() == 'true'

        prompt_configs = {
            'document': {
                'prompt': '<image>\n<|grounding|>Convert the document to markdown. ',
                'output_file': 'result.mmd'
            },
            'ocr': {
                'prompt': '<image>\n<|grounding|>OCR this image. ',
                'output_file': 'result.txt'
            },
            'free': {
                'prompt': '<image>\nFree OCR. ',
                'output_file': 'result.txt'
            },
            'figure': {
                'prompt': '<image>\nParse the figure. ',
                'output_file': 'result.txt'
            },
            'describe': {
                'prompt': '<image>\nDescribe this image in detail. ',
                'output_file': 'result.txt'
            }
        }
        config = prompt_configs.get(prompt_type, prompt_configs['document'])

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            pdf_file.save(tmp_pdf.name)
            temp_pdf_path = tmp_pdf.name

        doc = fitz.open(temp_pdf_path)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pages_output = []
        combined_texts = []

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=144)

            # 使用更健壮的临时文件处理方式，特别针对Windows系统
            page_dir = os.path.join(OUTPUT_DIR, f'pdf_page_{page_index+1}')
            os.makedirs(page_dir, exist_ok=True)

            # 直接在页面目录中创建唯一临时图像文件，避免与已存在文件冲突
            import uuid
            temp_image_path = os.path.join(page_dir, f"temp_page_{page_index+1}_{uuid.uuid4().hex}.jpg")

            try:
                # 保存图像到我们的临时位置
                pix.save(temp_image_path)

                update_progress('processing', 'ocr', f'Processing PDF page {page_index+1}/{len(doc)}', int( (page_index/ max(1,len(doc))) * 50))

                model.infer(
                    tokenizer,
                    prompt=config['prompt'],
                    image_file=temp_image_path,
                    output_path=page_dir,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=True,
                    test_compress=True
                )

                result_filepath = os.path.join(page_dir, config['output_file'])
                text = None
                if os.path.exists(result_filepath):
                    with open(result_filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    for filename in os.listdir(page_dir):
                        if filename.endswith(('.txt', '.mmd', '.md')):
                            with open(os.path.join(page_dir, filename), 'r', encoding='utf-8') as f:
                                text = f.read()
                            break

                boxes_image_rel = None
                boxes_image_path = os.path.join(page_dir, 'result_with_boxes.jpg')
                if os.path.exists(boxes_image_path):
                    boxes_image_rel = f'pdf_page_{page_index+1}/result_with_boxes.jpg'

                pages_output.append({
                    'page': page_index + 1,
                    'text': text or '',
                    'boxes_image_path': boxes_image_rel
                })
                combined_texts.append(text or '')
            finally:
                # 尝试删除临时图像文件，但不让错误中断处理
                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except (PermissionError, OSError) as e:
                        logger.warning(f"Could not remove temporary file {temp_image_path}: {e}")
                        # 在Windows上，有时文件会被锁定，我们稍后再尝试删除
                        import time
                        time.sleep(0.5)  # 等待500ms
                        try:
                            os.remove(temp_image_path)
                        except (PermissionError, OSError):
                            # 如果仍然失败，记录但继续处理
                            logger.warning(f"Still could not remove temporary file {temp_image_path} after retry")

        doc.close()
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

        update_progress('idle', '', '', 0, 0)

        return jsonify({
            'status': 'success',
            'prompt_type': prompt_type,
            'pages': pages_output,
            'combined_text': '\n\n'.join(combined_texts)
        })

    except Exception as e:
        logger.error(f"Error during PDF OCR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the model"""
    return jsonify({
        'model_name': MODEL_NAME,
        'cache_dir': MODEL_CACHE_DIR,
        'model_loaded': model is not None,
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'device_preference': device_preference
    })

@app.route('/outputs/<path:filename>', methods=['GET'])
def serve_output_file(filename):
    """Serve files from the outputs directory"""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/ocr_batch', methods=['POST'])
def perform_ocr_batch():
    """Perform OCR on multiple uploaded images"""
    global model, tokenizer

    try:
        if model is None or tokenizer is None:
            logger.info("Model not loaded, loading now...")
            if not load_model():
                return jsonify({'status': 'error', 'message': 'Failed to load model'}), 500

        images = request.files.getlist('images')
        if not images:
            return jsonify({'status': 'error', 'message': 'No images provided'}), 400

        prompt_type = request.form.get('prompt_type', 'document')
        base_size = int(request.form.get('base_size', 1024))
        image_size = int(request.form.get('image_size', 640))
        crop_mode = request.form.get('crop_mode', 'true').lower() == 'true'

        prompt_configs = {
            'document': {
                'prompt': '<image>\n<|grounding|>Convert the document to markdown. ',
                'output_file': 'result.mmd'
            },
            'ocr': {
                'prompt': '<image>\n<|grounding|>OCR this image. ',
                'output_file': 'result.txt'
            },
            'free': {
                'prompt': '<image>\nFree OCR. ',
                'output_file': 'result.txt'
            },
            'figure': {
                'prompt': '<image>\nParse the figure. ',
                'output_file': 'result.txt'
            },
            'describe': {
                'prompt': '<image>\nDescribe this image in detail. ',
                'output_file': 'result.txt'
            }
        }
        config = prompt_configs.get(prompt_type, prompt_configs['document'])

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        batch_results = []

        for idx, image_file in enumerate(images):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image_file.save(tmp_file.name)
                temp_image_path = tmp_file.name

            item_dir = os.path.join(OUTPUT_DIR, f'batch_{idx+1}')
            os.makedirs(item_dir, exist_ok=True)

            update_progress('processing', 'ocr', f'Processing image {idx+1}/{len(images)}', int((idx/ max(1,len(images))) * 50))

            model.infer(
                tokenizer,
                prompt=config['prompt'],
                image_file=temp_image_path,
                output_path=item_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=True,
                test_compress=True
            )

            result_filepath = os.path.join(item_dir, config['output_file'])
            text = None
            if os.path.exists(result_filepath):
                with open(result_filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                for filename in os.listdir(item_dir):
                    if filename.endswith(('.txt', '.mmd', '.md')):
                        with open(os.path.join(item_dir, filename), 'r', encoding='utf-8') as f:
                            text = f.read()
                        break

            boxes_image_rel = None
            boxes_image_path = os.path.join(item_dir, 'result_with_boxes.jpg')
            if os.path.exists(boxes_image_path):
                boxes_image_rel = f'batch_{idx+1}/result_with_boxes.jpg'

            batch_results.append({
                'index': idx + 1,
                'text': text or '',
                'boxes_image_path': boxes_image_rel
            })

            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

        update_progress('idle', '', '', 0, 0)

        combined_text = '\n\n'.join([item['text'] for item in batch_results])
        return jsonify({'status': 'success', 'prompt_type': prompt_type, 'items': batch_results, 'combined_text': combined_text})

    except Exception as e:
        logger.error(f"Error during batch OCR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting DeepSeek OCR Server...")
    logger.info("Model will be automatically downloaded on first use")

    # Suppress Flask's default request logging
    import logging as log
    log.getLogger('werkzeug').disabled = True

    # Run Flask server without request logging
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
