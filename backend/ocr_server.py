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
import json
from datetime import datetime
from threading import Thread, Lock
from pathlib import Path

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

# Queue processing state
processing_queue = []
queue_lock = Lock()
current_queue_id = None
queue_results = {}

# Use local cache directory relative to the app
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, '..', 'cache')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models')
OUTPUT_DIR = os.path.join(CACHE_DIR, 'outputs')

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
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir=MODEL_CACHE_DIR
            )
            logger.info("Using flash attention 2")
        except Exception as e:
            logger.warning(f"Flash attention not available: {e}, using default attention")
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir=MODEL_CACHE_DIR
            )

        # Stop download monitor and wait for it to finish
        download_monitor_active[0] = False
        monitor_thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish

        # Set to eval mode (80% progress)
        update_progress('loading', 'gpu', 'Moving model to GPU...', 80)
        model = model.eval()

        # Move to GPU if available with optimal dtype (85% progress)
        update_progress('loading', 'gpu', 'Optimizing model on GPU...', 85)
        if has_gpu:
            # Determine best dtype based on GPU capability
            compute_cap = torch.cuda.get_device_capability()
            if compute_cap[0] >= 8:  # Ampere or newer (RTX 30/40/50 series)
                model = model.cuda().to(torch.bfloat16)
                logger.info(f"Model loaded on GPU with bfloat16 (Compute {compute_cap[0]}.{compute_cap[1]})")
            else:  # Pascal/Turing (GTX 10/16 series, RTX 20 series)
                model = model.cuda().to(torch.float16)
                logger.info(f"Model loaded on GPU with float16 (Compute {compute_cap[0]}.{compute_cap[1]})")
        else:
            # CPU mode - use float32
            logger.info("Model loaded on CPU (inference will be slower)")

        # Apply torch.compile for ~30% inference speedup (PyTorch 2.0+) (95% progress)
        update_progress('loading', 'optimize', 'Compiling model with torch.compile...', 95)
        try:
            if hasattr(torch, 'compile') and has_gpu:
                logger.info("Applying torch.compile for faster inference...")
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled successfully (expect ~30% speedup)")
            else:
                logger.info("torch.compile not available or no GPU, skipping compilation")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}, using uncompiled model")

        # Warmup inference to initialize compiled graphs
        if has_gpu:
            update_progress('loading', 'warmup', 'Running warmup inference...', 98)
            logger.info("Running warmup inference...")
            try:
                # Create a small dummy image for warmup
                import numpy as np
                from PIL import Image
                dummy_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    dummy_img.save(tmp.name)
                    # Run a quick inference (won't save results)
                    logger.info("Warmup complete")
                    os.unlink(tmp.name)
            except Exception as e:
                logger.warning(f"Warmup inference failed: {e}")

        logger.info("Model loaded successfully!")
        update_progress('loaded', 'complete', 'Model ready!', 100)

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        update_progress('error', 'failed', str(e), 0)
        import traceback
        traceback.print_exc()

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
        return True

    # Start loading in background thread
    loading_thread = Thread(target=load_model_background)
    loading_thread.daemon = True
    loading_thread.start()

    return True

def clear_cuda_cache():
    """Clear CUDA cache to free memory between processing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def create_queue_output_folder():
    """Create a timestamped folder for queue processing results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    queue_folder = os.path.join(OUTPUT_DIR, f"queue_{timestamp}")
    os.makedirs(queue_folder, exist_ok=True)
    return queue_folder

def get_prompt_for_type(prompt_type):
    """Get prompt text for given type"""
    prompts = {
        'document': '<image>\n<|grounding|>Convert the document to markdown. ',
        'ocr': '<image>\n<|grounding|>OCR this image. ',
        'free': '<image>\nFree OCR. ',
        'figure': '<image>\nParse the figure. ',
        'describe': '<image>\nDescribe this image in detail. '
    }
    return prompts.get(prompt_type, prompts['document'])

def get_result_filename(prompt_type):
    """Get expected result filename for prompt type"""
    if prompt_type == 'document':
        return 'result.mmd'
    return 'result.txt'

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
                return jsonify({'status': 'error', 'message': 'Failed to load model'}), 500

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
                    except Exception:
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

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the model"""
    return jsonify({
        'model_name': MODEL_NAME,
        'cache_dir': MODEL_CACHE_DIR,
        'model_loaded': model is not None,
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })

@app.route('/queue/add', methods=['POST'])
def add_to_queue():
    """Add files to the processing queue"""
    global processing_queue
    
    try:
        # Get files from request
        files = request.files.getlist('files')
        if not files:
            return jsonify({'status': 'error', 'message': 'No files provided'}), 400
        
        # Get processing parameters
        prompt_type = request.form.get('prompt_type', 'document')
        base_size = int(request.form.get('base_size', 1024))
        image_size = int(request.form.get('image_size', 640))
        crop_mode = request.form.get('crop_mode', 'true').lower() == 'true'
        
        added_files = []
        with queue_lock:
            for file in files:
                if file.filename:
                    # Save file temporarily
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    file.save(temp_file.name)
                    temp_file.close()
                    
                    queue_item = {
                        'id': len(processing_queue),
                        'filename': file.filename,
                        'temp_path': temp_file.name,
                        'prompt_type': prompt_type,
                        'base_size': base_size,
                        'image_size': image_size,
                        'crop_mode': crop_mode,
                        'status': 'pending',
                        'progress': 0,
                        'result': None,
                        'error': None
                    }
                    processing_queue.append(queue_item)
                    added_files.append({'id': queue_item['id'], 'filename': file.filename})
        
        return jsonify({
            'status': 'success',
            'added': len(added_files),
            'files': added_files,
            'queue_length': len(processing_queue)
        })
    
    except Exception as e:
        logger.error(f"Error adding to queue: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/queue/status', methods=['GET'])
def get_queue_status():
    """Get current queue status with current file info for preview"""
    with queue_lock:
        # Find currently processing item
        current_item = next((item for item in processing_queue if item['status'] == 'processing'), None)
        
        queue_summary = {
            'total': len(processing_queue),
            'pending': sum(1 for item in processing_queue if item['status'] == 'pending'),
            'processing': sum(1 for item in processing_queue if item['status'] == 'processing'),
            'completed': sum(1 for item in processing_queue if item['status'] == 'completed'),
            'failed': sum(1 for item in processing_queue if item['status'] == 'failed'),
            'items': [{
                'id': item['id'],
                'filename': item['filename'],
                'status': item['status'],
                'progress': item['progress'],
                'error': item['error']
            } for item in processing_queue],
            'current_file': {
                'id': current_item['id'],
                'filename': current_item['filename'],
                'image_path': current_item.get('current_image_path'),
                'progress': current_item['progress']
            } if current_item else None
        }
    
    return jsonify(queue_summary)

@app.route('/queue/process', methods=['POST'])
def process_queue():
    """Start processing the queue sequentially"""
    global model, tokenizer, current_queue_id
    
    try:
        # Check if model is loaded, load it if not
        if model is None or tokenizer is None:
            logger.info("Model not loaded, loading now before processing queue...")
            load_model()
            
            # Wait for model to load (poll until loaded or error)
            max_wait = 300  # 5 minutes max
            start_time = time.time()
            while (time.time() - start_time) < max_wait:
                # Check progress status
                with progress_lock:
                    status = progress_data['status']
                    
                if status == 'loaded' and model is not None and tokenizer is not None:
                    logger.info("Model loaded successfully, starting queue processing")
                    break
                elif status == 'error':
                    error_msg = progress_data.get('message', 'Unknown error')
                    logger.error(f"Model loading failed: {error_msg}")
                    return jsonify({'status': 'error', 'message': f"Model loading failed: {error_msg}"}), 500
                
                time.sleep(2)  # Check every 2 seconds
            
            # Final check after timeout
            if model is None or tokenizer is None:
                logger.error("Model failed to load within timeout")
                return jsonify({'status': 'error', 'message': 'Model failed to load within 5 minutes. Please try loading model manually first.'}), 500
            
            # Extra safety: wait 2 more seconds for model to be fully ready
            logger.info("Waiting for model to be fully ready...")
            time.sleep(2)
        
        # Create output folder for this queue
        queue_folder = create_queue_output_folder()
        logger.info(f"Processing queue - output folder: {queue_folder}")
        
        # Process queue sequentially
        results_summary = []
        
        with queue_lock:
            items_to_process = [item for item in processing_queue if item['status'] == 'pending']
        
        for idx, item in enumerate(items_to_process):
            try:
                with queue_lock:
                    item['status'] = 'processing'
                    current_queue_id = item['id']
                
                progress_msg = f"[{idx + 1}/{len(items_to_process)}] Processing {item['filename']}"
                logger.info(f"=== {progress_msg} ===")
                
                # Create output subfolder for this file
                file_output_dir = os.path.join(queue_folder, f"file_{item['id']:03d}_{Path(item['filename']).stem}")
                os.makedirs(file_output_dir, exist_ok=True)
                
                # Update progress
                update_progress('processing', 'queue', progress_msg, int((idx / len(items_to_process)) * 100), 0)
                
                # Perform OCR with progress tracking (similar to single file)
                try:
                    # Store current file path for frontend to display
                    with queue_lock:
                        item['current_image_path'] = item['temp_path']
                    
                    # Use character counting stream like in perform_ocr
                    old_stdout = sys.stdout
                    char_count = [0]
                    
                    class CharCountingStream:
                        def __init__(self, original_stdout):
                            self.original = original_stdout
                            self.accumulated_text = ''
                            self.section_count = 0
                        
                        def write(self, text):
                            # Don't write to console (too verbose for queue)
                            # But do accumulate for progress tracking
                            self.accumulated_text += text
                            self.section_count = self.accumulated_text.count('=' * 20)
                            
                            if self.section_count >= 2:
                                parts = self.accumulated_text.split('=' * 20)
                                if len(parts) >= 3:
                                    raw_token_text = parts[2].strip().lstrip('=').strip()
                                    char_count[0] = len(raw_token_text)
                                    
                                    # Update progress with raw token stream
                                    if char_count[0] > 0:
                                        progress_msg = f"[{idx + 1}/{len(items_to_process)}] {item['filename']}"
                                        update_progress('processing', 'queue', progress_msg, 
                                                      int((idx / len(items_to_process)) * 100), 
                                                      char_count[0], raw_token_text)
                                        
                                        # Also update item progress
                                        with queue_lock:
                                            item['progress'] = min(int((char_count[0] / 1000) * 100), 90)
                        
                        def flush(self):
                            pass
                    
                    char_stream = CharCountingStream(old_stdout)
                    sys.stdout = char_stream
                    
                    try:
                        model.infer(
                            tokenizer,
                            prompt=get_prompt_for_type(item['prompt_type']),
                            image_file=item['temp_path'],
                            output_path=file_output_dir,
                            base_size=item['base_size'],
                            image_size=item['image_size'],
                            crop_mode=item['crop_mode'],
                            save_results=True,
                            test_compress=True
                        )
                    finally:
                        sys.stdout = old_stdout  # Restore stdout
                    
                    # Read result
                    result_file = get_result_filename(item['prompt_type'])
                    result_path = os.path.join(file_output_dir, result_file)
                    
                    result_text = None
                    if os.path.exists(result_path):
                        with open(result_path, 'r', encoding='utf-8') as f:
                            result_text = f.read()
                    
                    # Save metadata
                    metadata = {
                        'filename': item['filename'],
                        'prompt_type': item['prompt_type'],
                        'base_size': item['base_size'],
                        'image_size': item['image_size'],
                        'crop_mode': item['crop_mode'],
                        'processed_at': datetime.now().isoformat(),
                        'status': 'completed'
                    }
                    with open(os.path.join(file_output_dir, 'metadata.json'), 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    with queue_lock:
                        item['status'] = 'completed'
                        item['progress'] = 100
                        item['result'] = result_text
                    
                    logger.info(f"✓ Completed: {item['filename']} -> {file_output_dir}")
                    
                    results_summary.append({
                        'id': item['id'],
                        'filename': item['filename'],
                        'status': 'completed',
                        'output_dir': file_output_dir
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {item['filename']}: {e}")
                    with queue_lock:
                        item['status'] = 'failed'
                        item['error'] = str(e)
                    
                    results_summary.append({
                        'id': item['id'],
                        'filename': item['filename'],
                        'status': 'failed',
                        'error': str(e)
                    })
                
                finally:
                    # Clean up temp file
                    if os.path.exists(item['temp_path']):
                        os.remove(item['temp_path'])
                    
                    # Clear CUDA cache between items
                    clear_cuda_cache()
            
            except Exception as e:
                logger.error(f"Critical error processing queue item: {e}")
                with queue_lock:
                    item['status'] = 'failed'
                    item['error'] = str(e)
        
        # Save queue summary
        summary_path = os.path.join(queue_folder, 'queue_summary.json')
        completed_count = sum(1 for r in results_summary if r['status'] == 'completed')
        failed_count = sum(1 for r in results_summary if r['status'] == 'failed')
        
        with open(summary_path, 'w') as f:
            json.dump({
                'processed_at': datetime.now().isoformat(),
                'total_files': len(results_summary),
                'completed': completed_count,
                'failed': failed_count,
                'output_folder': queue_folder,
                'results': results_summary
            }, f, indent=2)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"QUEUE PROCESSING COMPLETE!")
        logger.info(f"Total: {len(results_summary)} | Completed: {completed_count} | Failed: {failed_count}")
        logger.info(f"Results saved to: {queue_folder}")
        logger.info("=" * 60)
        logger.info("")
        
        update_progress('idle', '', '', 0, 0)
        current_queue_id = None
        
        return jsonify({
            'status': 'success',
            'queue_folder': queue_folder,
            'completed': completed_count,
            'failed': failed_count,
            'total': len(results_summary),
            'results': results_summary
        })
    
    except Exception as e:
        logger.error(f"Error processing queue: {e}")
        update_progress('idle', '', '', 0, 0)
        current_queue_id = None
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/queue/clear', methods=['POST'])
def clear_queue():
    """Clear all items from the queue"""
    global processing_queue
    
    with queue_lock:
        # Clean up temp files
        for item in processing_queue:
            if 'temp_path' in item and os.path.exists(item['temp_path']):
                try:
                    os.remove(item['temp_path'])
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {item['temp_path']}: {e}")
        
        processing_queue.clear()
    
    return jsonify({'status': 'success', 'message': 'Queue cleared'})

@app.route('/queue/remove/<int:item_id>', methods=['DELETE'])
def remove_from_queue(item_id):
    """Remove a specific item from the queue"""
    global processing_queue
    
    with queue_lock:
        for i, item in enumerate(processing_queue):
            if item['id'] == item_id:
                # Clean up temp file
                if 'temp_path' in item and os.path.exists(item['temp_path']):
                    try:
                        os.remove(item['temp_path'])
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file: {e}")
                
                processing_queue.pop(i)
                return jsonify({'status': 'success', 'message': f'Item {item_id} removed'})
        
        return jsonify({'status': 'error', 'message': 'Item not found'}), 404

@app.route('/outputs/<path:filename>', methods=['GET'])
def serve_output_file(filename):
    """Serve files from the outputs directory"""
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting DeepSeek OCR Server...")
    logger.info("Model will be automatically downloaded on first use")

    # Suppress Flask's default request logging
    import logging as log
    log.getLogger('werkzeug').disabled = True

    # Run Flask server without request logging
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
