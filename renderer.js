const { ipcRenderer } = require('electron');

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const selectBtn = document.getElementById('select-btn');
const clearBtn = document.getElementById('clear-btn');
const viewBoxesBtn = document.getElementById('view-boxes-btn');
const viewTokensBtn = document.getElementById('view-tokens-btn');
const downloadZipBtn = document.getElementById('download-zip-btn');
const ocrBtn = document.getElementById('ocr-btn');
const ocrBtnText = document.getElementById('ocr-btn-text');
const pdfOcrBtn = document.getElementById('pdf-ocr-btn');
const loadModelBtn = document.getElementById('load-model-btn');
const copyBtn = document.getElementById('copy-btn');
const previewSection = document.getElementById('preview-section');
const imagePreview = document.getElementById('image-preview');
const resultsContent = document.getElementById('results-content');
const ocrPreviewImage = document.getElementById('ocr-preview-image');
const ocrBoxesOverlay = document.getElementById('ocr-boxes-overlay');
const progressInline = document.getElementById('progress-inline');
const progressStatus = document.getElementById('progress-status');
const serverUrlInput = document.getElementById('server-url');
const checkUpdatesBtn = document.getElementById('check-updates-btn');

// TTS elements
const ttsControls = document.getElementById('tts-controls');
const ttsEngineSelect = document.getElementById('tts-engine');
const readAloudBtn = document.getElementById('read-aloud-btn');
const stopReadBtn = document.getElementById('stop-read-btn');
const ttsAudio = document.getElementById('tts-audio');

// Lightbox elements
const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightbox-image');
const lightboxText = document.getElementById('lightbox-text');
const lightboxClose = document.querySelector('.lightbox-close');

// Status elements
const serverStatus = document.getElementById('server-status');
const modelStatus = document.getElementById('model-status');
const gpuStatus = document.getElementById('gpu-status');

// Form elements
const promptType = document.getElementById('prompt-type');
const baseSize = document.getElementById('base-size');
const imageSize = document.getElementById('image-size');
const cropMode = document.getElementById('crop-mode');
const ocrEngine = document.getElementById('ocr-engine');

// Constants
const DEEPSEEK_COORD_MAX = 999;

const KNOWN_TYPES = ['title', 'sub_title', 'text', 'table', 'image', 'image_caption', 'figure', 'caption', 'formula', 'list'];

const TYPE_COLORS = {
    'title': '#8B5CF6',
    'sub_title': '#A78BFA',
    'text': '#3B82F6',
    'table': '#F59E0B',
    'image': '#EC4899',
    'figure': '#06B6D4',
    'caption': '#10B981',
    'image_caption': '#4EC483',
    'formula': '#EF4444',
    'list': '#6366F1'
};

// State
let currentImagePath = null;
let currentResultText = null;
let currentRawTokens = null;
let currentPromptType = null;
let isProcessing = false;
let lastBoxCount = 0;

window.addEventListener('DOMContentLoaded', () => {
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            mangle: false,
            headerIds: false,
            breaks: true
        });
    }

    // Preload speech synthesis voices (they load asynchronously)
    if ('speechSynthesis' in window) {
        speechSynthesis.getVoices();
        speechSynthesis.onvoiceschanged = () => {
            const voices = speechSynthesis.getVoices();
            console.log('Speech synthesis voices loaded:', voices.length);
        };
    }

    checkServerStatus();
    setupEventListeners();
    setupTTS();
    setInterval(checkServerStatus, 5000);
});

function setupEventListeners() {
    // Image selection
    selectBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering dropZone click
        selectImage();
    });
    clearBtn.addEventListener('click', clearImage);
    viewBoxesBtn.addEventListener('click', viewBoxesImage);
    viewTokensBtn.addEventListener('click', viewRawTokens);
    imagePreview.addEventListener('click', viewOriginalImage);

    // Make entire drop zone clickable
    dropZone.addEventListener('click', selectImage);

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.background = '#e8eaff';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.background = '#f8f9ff';
    });
    dropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropZone.style.background = '#f8f9ff';

        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));

        if (files.length === 0) {
            showMessage('Please drop image file(s)', 'error');
            return;
        }

        if (files.length === 1) {
            // Single file - auto OCR
            await loadImage(files[0].path);
            performOCR();
        } else {
            // Multiple files - batch OCR
            const filePaths = files.map(f => f.path);
            await performBatchOCRWithPaths(filePaths);
        }
    });

    // OCR
    ocrBtn.addEventListener('click', performOCR);
    pdfOcrBtn.addEventListener('click', performPDFOCR);

    // Load model
    loadModelBtn.addEventListener('click', loadModel);

    // Copy results
    copyBtn.addEventListener('click', copyResults);

    // Download zip
    downloadZipBtn.addEventListener('click', downloadZip);

    // Lightbox
    lightboxClose.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) {
            closeLightbox();
        }
    });

    // Server URL persistence
    const savedUrl = localStorage.getItem('serverUrl');
    if (savedUrl) {
        serverUrlInput.value = savedUrl;
    }
    serverUrlInput.addEventListener('change', () => {
        localStorage.setItem('serverUrl', serverUrlInput.value.trim());
        checkServerStatus();
    });

    checkUpdatesBtn.addEventListener('click', checkForUpdates);

    // Engine selection
    ocrEngine.addEventListener('change', () => {
        checkServerStatus();
    });
}

function getServerUrl() {
    const url = (serverUrlInput && serverUrlInput.value) ? serverUrlInput.value.trim() : 'http://127.0.0.1:5000';
    return url.endsWith('/') ? url.slice(0, -1) : url;
}

async function checkServerStatus() {
    try {
        const result = await ipcRenderer.invoke('check-server-status', { serverUrl: getServerUrl() });

        if (result.success) {
            serverStatus.textContent = 'Connected';
            serverStatus.className = 'status-value success';

            const modelLoaded = result.data.model_loaded;
            modelStatus.textContent = modelLoaded ? 'Loaded' : 'Not loaded';
            modelStatus.className = `status-value ${modelLoaded ? 'success' : 'warning'}`;

            const gpuAvailable = result.data.gpu_available;
            const tesseractAvailable = result.data.tesseract_available;
            const deepseekAvailable = result.data.deepseek_available;
            const currentEngine = ocrEngine ? ocrEngine.value : 'tesseract';

            // Update GPU status based on engine
            if (currentEngine === 'tesseract') {
                gpuStatus.textContent = tesseractAvailable ? 'Tesseract OK' : 'Tesseract Missing';
                gpuStatus.className = `status-value ${tesseractAvailable ? 'success' : 'error'}`;
            } else {
                gpuStatus.textContent = gpuAvailable ? 'GPU Available' : 'No GPU';
                gpuStatus.className = `status-value ${gpuAvailable ? 'success' : 'error'}`;
            }

            // Update load model button state (but don't change if currently processing)
            if (!isProcessing) {
                if (modelLoaded) {
                    loadModelBtn.disabled = true;
                    loadModelBtn.textContent = 'Model Loaded ✓';
                    loadModelBtn.classList.add('btn-loaded');
                } else {
                    loadModelBtn.disabled = false;
                    loadModelBtn.textContent = 'Load Model';
                    loadModelBtn.classList.remove('btn-loaded');
                }
            }

            // Update OCR button state - only enable if both image loaded AND model loaded (and not currently processing)
            if (!isProcessing) {
                if (currentImagePath && modelLoaded) {
                    ocrBtn.disabled = false;
                } else {
                    ocrBtn.disabled = true;
                }
            }
        } else {
            serverStatus.textContent = 'Disconnected';
            serverStatus.className = 'status-value error';
            modelStatus.textContent = 'Unknown';
            modelStatus.className = 'status-value';
            gpuStatus.textContent = 'Unknown';
            gpuStatus.className = 'status-value';

            // Disable OCR if server disconnected
            ocrBtn.disabled = true;
        }
    } catch (error) {
        console.error('Status check error:', error);
    }
}

async function selectImage() {
    const result = await ipcRenderer.invoke('select-images');

    if (result.success && result.filePaths && result.filePaths.length > 0) {
        if (result.filePaths.length === 1) {
            // Single file - auto OCR
            await loadImage(result.filePaths[0]);
            performOCR();
        } else {
            // Multiple files - batch OCR direct start
            await performBatchOCRWithPaths(result.filePaths);
        }
    }
}

async function loadImage(filePath) {
    currentImagePath = filePath;
    imagePreview.src = filePath;

    dropZone.style.display = 'none';
    previewSection.style.display = 'block';

    // Clear previous results
    ocrPreviewImage.src = '';
    resultsContent.innerHTML = '';
    progressInline.style.display = 'none';
    copyBtn.style.display = 'none';
    copyBtn.style.display = 'none';
    downloadZipBtn.style.display = 'none';
    if (ttsControls) ttsControls.style.display = 'none';
    viewBoxesBtn.style.display = 'none';
    viewTokensBtn.style.display = 'none';

    // Clear overlay boxes
    ocrBoxesOverlay.innerHTML = '';
    ocrBoxesOverlay.removeAttribute('viewBox');
    lastBoxCount = 0;

    // Check server status to update OCR button state
    await checkServerStatus();
}

function clearImage() {
    currentImagePath = null;
    currentResultText = null;
    currentRawTokens = null;
    currentPromptType = null;
    imagePreview.src = '';

    dropZone.style.display = 'block';
    previewSection.style.display = 'none';
    ocrBtn.disabled = true;
    viewBoxesBtn.style.display = 'none';
    viewTokensBtn.style.display = 'none';

    // Clear results and progress
    ocrPreviewImage.src = '';
    resultsContent.innerHTML = '';
    progressInline.style.display = 'none';
    copyBtn.style.display = 'none';
    downloadZipBtn.style.display = 'none';

    // Clear overlay boxes
    ocrBoxesOverlay.innerHTML = '';
    ocrBoxesOverlay.removeAttribute('viewBox');
    lastBoxCount = 0;
}

function openLightbox(imageSrc) {
    lightboxImage.src = imageSrc;
    lightboxImage.style.display = 'block';
    lightboxText.style.display = 'none';
    lightbox.style.display = 'block';
}

function openLightboxWithText(text) {
    lightboxText.textContent = text;
    lightboxText.style.display = 'block';
    lightboxImage.style.display = 'none';
    lightbox.style.display = 'block';
}

function closeLightbox() {
    lightbox.style.display = 'none';
}

function viewOriginalImage() {
    if (currentImagePath) {
        openLightbox(currentImagePath);
    }
}

async function viewBoxesImage() {
    if (!currentImagePath) return;

    // Create a canvas to render the image with boxes
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Load the original image
    const img = new Image();
    img.src = currentImagePath;

    await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
    });

    // Set canvas size to match image
    canvas.width = img.width;
    canvas.height = img.height;

    // Draw the original image
    ctx.drawImage(img, 0, 0);

    // Parse boxes from current raw tokens
    if (currentRawTokens) {
        const boxes = parseBoxesFromTokens(currentRawTokens, true); // OCR is complete when viewing boxes

        // Helper to convert hex to rgba
        const hexToRgba = (hex, alpha) => {
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        };

        // Draw each box
        boxes.forEach((box) => {
            const x1 = (box.x1 / DEEPSEEK_COORD_MAX) * img.width;
            const y1 = (box.y1 / DEEPSEEK_COORD_MAX) * img.height;
            const x2 = (box.x2 / DEEPSEEK_COORD_MAX) * img.width;
            const y2 = (box.y2 / DEEPSEEK_COORD_MAX) * img.height;

            const color = TYPE_COLORS[box.type] || '#FF1493';
            const isUnknownType = !TYPE_COLORS[box.type];

            // Draw semi-transparent fill
            ctx.fillStyle = isUnknownType ? 'rgba(0, 255, 0, 0.3)' : hexToRgba(color, 0.1);
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);

            // Draw border
            ctx.strokeStyle = color;
            ctx.lineWidth = isUnknownType ? 3 : 2;
            ctx.globalAlpha = 0.9;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.globalAlpha = 1.0;

            // Draw label
            const labelPadding = 4;
            const labelHeight = 18;
            const displayText = box.isType ? box.type : (box.content.length > 30 ? box.content.substring(0, 30) + '...' : box.content);
            ctx.font = isUnknownType ? 'bold 12px system-ui' : '500 12px system-ui';
            const labelWidth = ctx.measureText(displayText).width + labelPadding * 2;
            const labelY = Math.max(0, y1 - labelHeight);

            // Label background
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.95;
            ctx.fillRect(x1, labelY, labelWidth, labelHeight);
            ctx.globalAlpha = 1.0;

            // Label text
            ctx.fillStyle = isUnknownType ? '#00FF00' : 'white';
            ctx.fillText(displayText, x1 + labelPadding, labelY + 13);
        });
    }

    // Convert canvas to image and show in lightbox
    const imageUrl = canvas.toDataURL('image/png');
    openLightbox(imageUrl);
}

function viewRawTokens() {
    if (currentRawTokens) {
        openLightboxWithText(currentRawTokens);
    }
}

function parseBoxesFromTokens(tokenText, isOcrComplete = false) {
    // Extract all bounding boxes from token format: <|ref|>CONTENT<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
    const boxes = [];
    const refDetRegex = /<\|ref\|>([^<]+)<\|\/ref\|><\|det\|>\[\[([^\]]+)\]\]<\|\/det\|>/g;
    let match;
    const matches = [];

    // First, collect all matches with their positions
    while ((match = refDetRegex.exec(tokenText)) !== null) {
        matches.push({
            content: match[1].trim(),
            coords: match[2],
            matchStart: match.index,
            matchEnd: match.index + match[0].length
        });
    }

    // Now process each match and determine if it's complete
    for (let i = 0; i < matches.length; i++) {
        try {
            const matchData = matches[i];
            const content = matchData.content;

            // Parse the coordinate string "x1,\ny1,\nx2,\ny2" or "x1, y1, x2, y2"
            const coords = matchData.coords.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
            if (coords.length === 4) {
                // Determine if this is a type label or actual text content
                const isType = KNOWN_TYPES.includes(content);

                // Extract the actual text content that comes after this box (for Document mode)
                let textContent = '';
                let isComplete = false;

                if (i < matches.length - 1) {
                    // Not the last box - extract content between this box and the next
                    textContent = tokenText.substring(matchData.matchEnd, matches[i + 1].matchStart).trim();
                    isComplete = textContent.length > 0;
                } else {
                    // Last box - extract everything after it
                    textContent = tokenText.substring(matchData.matchEnd).trim();
                    isComplete = isOcrComplete && textContent.length > 0;
                }

                boxes.push({
                    content: content,
                    textContent: textContent,  // The actual text to copy in Document mode
                    isType: isType,
                    type: isType ? content : 'text',  // Use 'text' as default type for OCR content
                    x1: coords[0],
                    y1: coords[1],
                    x2: coords[2],
                    y2: coords[3],
                    isComplete: isComplete  // Add completion status for Document mode
                });
            }
        } catch (e) {
            console.error('Error parsing box coordinates:', e);
        }
    }

    return boxes;
}

function extractTextFromTokens(tokenText) {
    // Extract just the text content (non-type labels) from tokens
    const boxes = parseBoxesFromTokens(tokenText);
    const textPieces = boxes
        .filter(box => !box.isType)  // Only non-type content
        .map(box => box.content);
    return textPieces.join('\n');  // Join with newlines for readability
}

function renderBoxes(boxes, imageWidth, imageHeight, promptType) {
    if (!imageWidth || !imageHeight || boxes.length === 0) {
        return;
    }

    // Set SVG viewBox to match image dimensions (only once)
    if (!ocrBoxesOverlay.hasAttribute('viewBox')) {
        ocrBoxesOverlay.setAttribute('viewBox', `0 0 ${imageWidth} ${imageHeight}`);
        ocrBoxesOverlay.setAttribute('preserveAspectRatio', 'none');
    }

    // OCR Text and Document modes have interactive boxes
    const isInteractive = promptType === 'ocr' || promptType === 'document';

    // Only add new boxes that haven't been rendered yet
    const newBoxes = boxes.slice(lastBoxCount);

    newBoxes.forEach((box) => {
        // Scale coordinates from 0-999 normalized space to actual image dimensions
        const scaledX1 = (box.x1 / DEEPSEEK_COORD_MAX) * imageWidth;
        const scaledY1 = (box.y1 / DEEPSEEK_COORD_MAX) * imageHeight;
        const scaledX2 = (box.x2 / DEEPSEEK_COORD_MAX) * imageWidth;
        const scaledY2 = (box.y2 / DEEPSEEK_COORD_MAX) * imageHeight;

        // Get color for this box type - use bright pink/green if unknown
        const color = TYPE_COLORS[box.type] || '#FF1493';  // Hot pink for unknown types
        const isUnknownType = !TYPE_COLORS[box.type];

        // Create group for box and label
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('class', 'ocr-box-group');
        group.style.cursor = box.isType ? 'default' : 'pointer';

        // Create semi-transparent fill rectangle
        const fillRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        fillRect.setAttribute('x', scaledX1);
        fillRect.setAttribute('y', scaledY1);
        fillRect.setAttribute('width', scaledX2 - scaledX1);
        fillRect.setAttribute('height', scaledY2 - scaledY1);
        fillRect.setAttribute('fill', isUnknownType ? '#00FF00' : color);  // Lime green for unknown
        fillRect.setAttribute('opacity', isUnknownType ? '0.3' : '0.1');
        fillRect.setAttribute('class', 'ocr-box-fill');

        // Create border rectangle
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', scaledX1);
        rect.setAttribute('y', scaledY1);
        rect.setAttribute('width', scaledX2 - scaledX1);
        rect.setAttribute('height', scaledY2 - scaledY1);
        rect.setAttribute('fill', 'none');
        rect.setAttribute('stroke', color);  // Hot pink border for unknown
        rect.setAttribute('stroke-width', isUnknownType ? '3' : '2');
        rect.setAttribute('opacity', '0.9');
        rect.setAttribute('class', 'ocr-box-border');

        // Create label background
        const labelBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        const labelPadding = 4;
        const labelHeight = 18;
        const displayText = box.isType ? box.type : (box.content.length > 30 ? box.content.substring(0, 30) + '...' : box.content);
        const labelWidth = displayText.length * 7 + labelPadding * 2;

        labelBg.setAttribute('x', scaledX1);
        labelBg.setAttribute('y', Math.max(0, scaledY1 - labelHeight));
        labelBg.setAttribute('width', labelWidth);
        labelBg.setAttribute('height', labelHeight);
        labelBg.setAttribute('fill', color);  // Hot pink background for unknown types
        labelBg.setAttribute('opacity', '0.95');
        labelBg.setAttribute('class', 'ocr-box-label-bg');

        // Create label text
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', scaledX1 + labelPadding);
        label.setAttribute('y', Math.max(0, scaledY1 - labelHeight) + 13);
        label.setAttribute('fill', isUnknownType ? '#00FF00' : 'white');  // Lime green text for unknown
        label.setAttribute('font-size', '12');
        label.setAttribute('font-family', 'system-ui, -apple-system, sans-serif');
        label.setAttribute('font-weight', isUnknownType ? '700' : '500');
        label.setAttribute('class', 'ocr-box-label-text');
        label.textContent = displayText;

        // Add hover and click interactions
        // OCR mode: text content boxes (not type labels) are clickable
        // Document mode: type label boxes with complete text content are clickable
        let isClickable = false;
        let copyText = '';

        if (promptType === 'ocr') {
            // OCR mode: clickable if not a type label
            isClickable = !box.isType && isInteractive;
            copyText = box.content;
        } else if (promptType === 'document') {
            // Document mode: clickable if it's a type label with complete text content
            isClickable = box.isType && box.isComplete && box.textContent && isInteractive;
            copyText = box.textContent;
        }

        if (isClickable) {
            // Enable pointer events
            group.style.pointerEvents = 'all';
            group.style.cursor = 'pointer';

            group.addEventListener('mouseenter', (e) => {
                fillRect.setAttribute('opacity', '0.3');
                rect.setAttribute('stroke-width', isUnknownType ? '4' : '3');
                labelBg.setAttribute('opacity', '1');
                e.stopPropagation();
            });

            group.addEventListener('mouseleave', (e) => {
                fillRect.setAttribute('opacity', isUnknownType ? '0.3' : '0.1');
                rect.setAttribute('stroke-width', isUnknownType ? '3' : '2');
                labelBg.setAttribute('opacity', '0.95');
                e.stopPropagation();
            });

            group.addEventListener('click', async (e) => {
                e.stopPropagation();
                try {
                    await navigator.clipboard.writeText(copyText);
                    // Visual feedback - flash the label
                    const originalBg = labelBg.getAttribute('fill');
                    const originalText = label.textContent;
                    labelBg.setAttribute('fill', '#10B981');  // Green
                    label.textContent = '✓ Copied!';
                    console.log('Copied text:', copyText);
                    setTimeout(() => {
                        labelBg.setAttribute('fill', originalBg);
                        label.textContent = originalText;
                    }, 1000);
                } catch (err) {
                    console.error('Failed to copy text:', err);
                }
            });

            // Add title for tooltip
            const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            const previewText = copyText.length > 50 ? copyText.substring(0, 50) + '...' : copyText;
            title.textContent = `Click to copy: ${previewText}`;
            group.appendChild(title);
        } else if (box.isType && promptType === 'document' && !box.isComplete) {
            // Document mode incomplete boxes - show as non-clickable but with visual feedback
            group.style.pointerEvents = 'none';
            group.style.cursor = 'default';
            group.style.opacity = '0.6';  // Dimmed to show it's not ready yet
        } else {
            // Non-interactive boxes
            group.style.pointerEvents = 'none';
        }

        // Add animation for new boxes
        group.style.animation = 'fadeIn 0.3s ease-in';

        group.appendChild(fillRect);
        group.appendChild(rect);
        group.appendChild(labelBg);
        group.appendChild(label);

        ocrBoxesOverlay.appendChild(group);
    });

    // Update the count of rendered boxes
    lastBoxCount = boxes.length;
}

async function loadModel() {
    if (isProcessing) return;

    let pollInterval = null;

    try {
        isProcessing = true;
        loadModelBtn.disabled = true;
        loadModelBtn.textContent = 'Loading Model...';

        const modelProgress = document.getElementById('model-progress');
        const progressBar = document.getElementById('progress-bar');
        const progressStage = document.getElementById('progress-stage');
        const progressPercent = document.getElementById('progress-percent');
        modelProgress.style.display = 'block';

        // Start polling for progress updates
        const pollProgress = async () => {
            try {
                const response = await fetch(`${getServerUrl()}/progress`);
                const data = await response.json();
                console.log('Progress update:', data);

                if (data.status === 'loading') {
                    const percent = Math.max(0, Math.min(100, data.progress_percent || 0));
                    progressBar.style.width = `${percent}%`;
                    progressStage.textContent = data.stage || '';
                    progressPercent.textContent = `${percent}%`;
                } else if (data.status === 'loaded') {
                    progressBar.style.width = '100%';
                    progressStage.textContent = 'complete';
                    progressPercent.textContent = '100%';

                    // Stop polling when done
                    if (pollInterval) {
                        clearInterval(pollInterval);
                        pollInterval = null;
                    }
                } else if (data.status === 'error') {
                    progressStage.textContent = 'error';

                    // Stop polling on error
                    if (pollInterval) {
                        clearInterval(pollInterval);
                        pollInterval = null;
                    }
                }
            } catch (error) {
                console.error('Error polling progress:', error);
            }
        };

        // Poll every 400ms
        pollInterval = setInterval(pollProgress, 500);

        // Trigger model loading
        const currentEngine = ocrEngine ? ocrEngine.value : 'tesseract';
        const result = await ipcRenderer.invoke('load-model', { serverUrl: getServerUrl(), ocr_engine: currentEngine });

        // Wait for final status
        await new Promise(resolve => {
            const checkStatus = setInterval(async () => {
                try {
                    const response = await fetch(`${getServerUrl()}/progress`);
                    const data = await response.json();

                    if (data.status === 'loaded' || data.status === 'error') {
                        clearInterval(checkStatus);
                        if (pollInterval) {
                            clearInterval(pollInterval);
                            pollInterval = null;
                        }
                        resolve();
                    }
                } catch (error) {
                    console.error('Error checking status:', error);
                }
            }, 500);
        });

        modelProgress.style.display = 'none';

        if (result.success) {
            showMessage('Model loaded successfully!', 'success');
            await checkServerStatus();
        } else {
            showMessage(`Failed to load model: ${result.error}`, 'error');
            await checkServerStatus(); // Update button state even on failure
        }
    } catch (error) {
        if (pollInterval) {
            clearInterval(pollInterval);
        }
        progressInline.style.display = 'none';
        showMessage(`Error: ${error.message}`, 'error');
        await checkServerStatus(); // Update button state even on error
    } finally {
        if (pollInterval) {
            clearInterval(pollInterval);
        }
        isProcessing = false;
        // Don't reset button state here - let checkServerStatus() handle it
    }
}

async function performOCR() {
    if (!currentImagePath || isProcessing) return;

    let tokenPollInterval = null;
    let imageNaturalWidth = 0;
    let imageNaturalHeight = 0;

    try {
        isProcessing = true;
        ocrBtn.disabled = true;
        ocrBtnText.textContent = 'Processing...';

        // Store current prompt type
        currentPromptType = promptType.value;

        // Show progress in header
        progressInline.style.display = 'flex';
        progressStatus.textContent = 'Starting OCR...';

        // Clear panels
        resultsContent.innerHTML = '';
        copyBtn.style.display = 'none';

        // Reset box tracking
        lastBoxCount = 0;
        ocrBoxesOverlay.innerHTML = '';
        ocrBoxesOverlay.removeAttribute('viewBox');

        // Load image into preview and get dimensions
        ocrPreviewImage.src = currentImagePath;
        await new Promise((resolve) => {
            ocrPreviewImage.onload = () => {
                imageNaturalWidth = ocrPreviewImage.naturalWidth;
                imageNaturalHeight = ocrPreviewImage.naturalHeight;
                console.log(`Image dimensions: ${imageNaturalWidth}×${imageNaturalHeight}`);
                resolve();
            };
        });

        // Poll for token count and raw token stream updates
        tokenPollInterval = setInterval(async () => {
            try {
                const response = await fetch(`${getServerUrl()}/progress`);
                const data = await response.json();

                if (data.status === 'processing') {
                    if (data.chars_generated > 0) {
                        progressStatus.textContent = `${data.chars_generated} characters generated`;
                    }

                    // Parse and render boxes from raw token stream
                    if (data.raw_token_stream) {
                        const boxes = parseBoxesFromTokens(data.raw_token_stream, false); // Still streaming, not complete
                        renderBoxes(boxes, imageNaturalWidth, imageNaturalHeight, currentPromptType);

                        // Update text panel in real-time
                        if (currentPromptType === 'ocr') {
                            // OCR mode: show extracted text
                            const extractedText = extractTextFromTokens(data.raw_token_stream);
                            if (extractedText) {
                                resultsContent.textContent = extractedText;
                            }
                        } else if (currentPromptType === 'document') {
                            // Document mode: show raw markdown (will be rendered later)
                            resultsContent.textContent = data.raw_token_stream;
                        } else {
                            // Free OCR, Figure, Describe modes: show raw tokens streaming
                            resultsContent.textContent = data.raw_token_stream;
                        }
                    }
                }
            } catch (error) {
                // Ignore polling errors
            }
        }, 200); // Poll every 200ms for smooth updates

        const result = await ipcRenderer.invoke('perform-ocr', {
            imagePath: currentImagePath,
            promptType: promptType.value,
            baseSize: parseInt(baseSize.value),
            imageSize: parseInt(imageSize.value),
            cropMode: cropMode.checked,
            serverUrl: getServerUrl()
        });

        // Stop polling
        if (tokenPollInterval) {
            clearInterval(tokenPollInterval);
            tokenPollInterval = null;
        }

        if (result.success) {
            // Hide progress spinner
            progressInline.style.display = 'none';

            // Store raw tokens
            currentRawTokens = result.data.raw_tokens;

            // Do a final render of all boxes with the complete token stream
            // This ensures any boxes that arrived after polling stopped are rendered
            if (currentRawTokens && imageNaturalWidth && imageNaturalHeight) {
                const boxes = parseBoxesFromTokens(currentRawTokens, true); // OCR is complete
                // Reset lastBoxCount to 0 to force re-render of all boxes
                lastBoxCount = 0;
                ocrBoxesOverlay.innerHTML = '';
                renderBoxes(boxes, imageNaturalWidth, imageNaturalHeight, currentPromptType);
            }

            // Display results based on mode
            if (result.data.prompt_type === 'ocr') {
                // OCR Text mode: extract and show just the text
                const extractedText = currentRawTokens ? extractTextFromTokens(currentRawTokens) : result.data.result;
                resultsContent.textContent = extractedText;
                currentResultText = extractedText;
            } else if (result.data.prompt_type === 'document') {
                // Document mode: render markdown
                displayResults(result.data.result, result.data.prompt_type);
                currentResultText = result.data.result;
            } else {
                // Free OCR, Figure, Describe modes: show raw tokens
                const rawText = currentRawTokens || result.data.result;
                resultsContent.textContent = rawText;
                currentResultText = rawText;
            }

            // Always show copy button when we have results
            copyBtn.style.display = 'inline-block';

            // Show download zip button only for document mode
            if (currentPromptType === 'document') {
                downloadZipBtn.style.display = 'inline-block';
            } else {
                downloadZipBtn.style.display = 'none';
            }

            // Show raw tokens button and boxes button if raw tokens exist
            if (currentRawTokens) {
                viewTokensBtn.style.display = 'inline-block';
                viewBoxesBtn.style.display = 'inline-block';
            } else {
                viewTokensBtn.style.display = 'none';
                viewBoxesBtn.style.display = 'none';
            }

            showMessage('OCR completed successfully!', 'success');
        } else {
            // Error handling
            ocrBoxesOverlay.innerHTML = '';
            ocrBoxesOverlay.removeAttribute('viewBox');
            lastBoxCount = 0;

            ocrPreviewImage.src = '';
            progressInline.style.display = 'none';
            resultsContent.innerHTML = `<p class="error">Error: ${result.error}</p>`;
            copyBtn.style.display = 'none';
            downloadZipBtn.style.display = 'none';
            viewBoxesBtn.style.display = 'none';
            viewTokensBtn.style.display = 'none';
            showMessage(`OCR failed: ${result.error}`, 'error');
        }
    } catch (error) {
        if (tokenPollInterval) {
            clearInterval(tokenPollInterval);
        }
        ocrBoxesOverlay.innerHTML = '';
        ocrBoxesOverlay.removeAttribute('viewBox');
        lastBoxCount = 0;
        ocrPreviewImage.src = '';
        progressInline.style.display = 'none';
        resultsContent.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        copyBtn.style.display = 'none';
        downloadZipBtn.style.display = 'none';
        viewBoxesBtn.style.display = 'none';
        viewTokensBtn.style.display = 'none';
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        if (tokenPollInterval) {
            clearInterval(tokenPollInterval);
        }
        isProcessing = false;
        ocrBtnText.textContent = 'Run OCR';
        // Check server status to properly set button state based on model loaded status
        await checkServerStatus();
    }
}

function displayResults(result, promptType) {
    // Format the result nicely
    let formattedResult = '';

    if (typeof result === 'string') {
        formattedResult = result;
    } else if (typeof result === 'object') {
        formattedResult = JSON.stringify(result, null, 2);
    } else {
        formattedResult = String(result);
    }

    // Store original text for copying (with relative paths)
    currentResultText = formattedResult;

    // Render markdown for document mode
    if (promptType === 'document' && typeof marked !== 'undefined') {
        const cacheBuster = Date.now();
        const renderedMarkdown = formattedResult.replace(
            /!\[([^\]]*)\]\(images\/([^)]+)\)/g,
            `![$1](${getServerUrl()}/outputs/images/$2?t=${cacheBuster})`
        );
        resultsContent.innerHTML = marked.parse(renderedMarkdown);
    } else {
        resultsContent.textContent = formattedResult;
    }

    // Show TTS controls
    if (formattedResult && ttsControls) {
        console.log('Showing TTS controls');
        ttsControls.style.display = 'flex';
        readAloudBtn.style.display = 'inline-block';
        stopReadBtn.style.display = 'none';
        if (speechSynthesis.speaking) speechSynthesis.cancel();

    }
}



function copyResults() {
    // Use the original text (markdown) instead of rendered HTML
    const text = currentResultText || resultsContent.textContent;

    navigator.clipboard.writeText(text).then(() => {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        setTimeout(() => {
            copyBtn.textContent = originalText;
        }, 2000);
    }).catch(err => {
        showMessage('Failed to copy to clipboard', 'error');
    });
}

async function downloadZip() {
    if (!currentResultText || currentPromptType !== 'document') {
        showMessage('No document to download', 'error');
        return;
    }

    try {
        // Show loading state
        const originalText = downloadZipBtn.textContent;
        downloadZipBtn.textContent = 'Creating ZIP...';
        downloadZipBtn.disabled = true;

        // Create a new JSZip instance
        const zip = new JSZip();

        // Add the markdown file
        zip.file('output.md', currentResultText);

        // Find all image references in the markdown
        const imageRegex = /!\[([^\]]*)\]\(images\/([^)]+)\)/g;
        const imageFiles = new Set();
        let match;

        while ((match = imageRegex.exec(currentResultText)) !== null) {
            imageFiles.add(match[2]); // Extract filename like "0.jpg"
        }

        // Fetch and add each image to the zip
        const imagesFolder = zip.folder('images');
        const imagePromises = Array.from(imageFiles).map(async (filename) => {
            try {
                const response = await fetch(`${getServerUrl()}/outputs/images/${filename}`);
                if (response.ok) {
                    const blob = await response.blob();
                    imagesFolder.file(filename, blob);
                } else {
                    console.warn(`Failed to fetch image: ${filename}`);
                }
            } catch (error) {
                console.error(`Error fetching image ${filename}:`, error);
            }
        });

        // Wait for all images to be fetched
        await Promise.all(imagePromises);

        // Generate the zip file
        const zipBlob = await zip.generateAsync({ type: 'blob' });

        // Create download link and trigger download
        const url = URL.createObjectURL(zipBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ocr-output-${Date.now()}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Reset button state
        downloadZipBtn.textContent = 'Downloaded!';
        setTimeout(() => {
            downloadZipBtn.textContent = originalText;
            downloadZipBtn.disabled = false;
        }, 2000);

        showMessage('ZIP file downloaded successfully', 'success');
    } catch (error) {
        console.error('Error creating ZIP:', error);
        showMessage('Failed to create ZIP file', 'error');
        downloadZipBtn.textContent = 'Download ZIP';
        downloadZipBtn.disabled = false;
    }
}

function showMessage(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    if (resultsContent.textContent.includes('OCR results will appear here')) {
        resultsContent.innerHTML = `<p class="${type}">${message}</p>`;
    }
}
async function performPDFOCR() {
    if (isProcessing) return;

    try {
        isProcessing = true;
        pdfOcrBtn.disabled = true;
        progressInline.style.display = 'flex';
        progressStatus.textContent = 'Selecting PDF...';

        const sel = await ipcRenderer.invoke('select-pdf');
        if (!sel.success) {
            progressInline.style.display = 'none';
            pdfOcrBtn.disabled = false;
            return;
        }

        progressStatus.textContent = 'Processing PDF...';

        const result = await ipcRenderer.invoke('perform-ocr-pdf', {
            pdfPath: sel.filePath,
            promptType: promptType.value,
            baseSize: parseInt(baseSize.value),
            imageSize: parseInt(imageSize.value),
            cropMode: cropMode.checked,
            serverUrl: getServerUrl()
        });

        progressInline.style.display = 'none';

        if (result.success) {
            currentPromptType = result.data.prompt_type;
            const combinedText = result.data.combined_text || '';
            displayResults(combinedText, currentPromptType);
            copyBtn.style.display = 'inline-block';
            downloadZipBtn.style.display = currentPromptType === 'document' ? 'inline-block' : 'none';
            showMessage('PDF OCR completed successfully', 'success');
        } else {
            resultsContent.innerHTML = `<p class="error">Error: ${result.error}</p>`;
            copyBtn.style.display = 'none';
            downloadZipBtn.style.display = 'none';
            showMessage(`PDF OCR failed: ${result.error}`, 'error');
        }
    } catch (error) {
        progressInline.style.display = 'none';
        resultsContent.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        copyBtn.style.display = 'none';
        downloadZipBtn.style.display = 'none';
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        isProcessing = false;
        pdfOcrBtn.disabled = false;
    }
}
async function performBatchOCRWithPaths(imagePaths) {
    if (isProcessing) return;
    if (!imagePaths || imagePaths.length === 0) return;

    try {
        isProcessing = true;
        progressInline.style.display = 'flex';
        progressStatus.textContent = `Processing ${imagePaths.length} images...`;

        // Clear previous results
        resultsContent.innerHTML = '';
        ocrPreviewImage.src = '';
        ocrBoxesOverlay.innerHTML = '';
        ocrBoxesOverlay.removeAttribute('viewBox');
        lastBoxCount = 0;
        copyBtn.style.display = 'none';
        downloadZipBtn.style.display = 'none';
        viewBoxesBtn.style.display = 'none';
        viewTokensBtn.style.display = 'none';

        // Hide drop zone, show preview section with batch info
        dropZone.style.display = 'none';
        previewSection.style.display = 'block';
        imagePreview.src = imagePaths[0]; // Show first image as preview

        const currentEngine = ocrEngine ? ocrEngine.value : 'tesseract';

        const result = await ipcRenderer.invoke('perform-ocr-batch', {
            imagePaths: imagePaths,
            promptType: promptType.value,
            baseSize: parseInt(baseSize.value),
            imageSize: parseInt(imageSize.value),
            cropMode: cropMode.checked,
            serverUrl: getServerUrl(),
            ocr_engine: currentEngine
        });

        progressInline.style.display = 'none';

        if (result.success) {
            currentPromptType = result.data.prompt_type;
            const combinedText = result.data.combined_text || '';
            displayResults(combinedText, currentPromptType);
            copyBtn.style.display = 'inline-block';
            downloadZipBtn.style.display = currentPromptType === 'document' ? 'inline-block' : 'none';
            showMessage(`Batch OCR completed - ${imagePaths.length} images processed`, 'success');
        } else {
            resultsContent.innerHTML = `<p class="error">Error: ${result.error}</p>`;
            copyBtn.style.display = 'none';
            downloadZipBtn.style.display = 'none';
            showMessage(`Batch OCR failed: ${result.error}`, 'error');
        }
    } catch (error) {
        progressInline.style.display = 'none';
        resultsContent.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        copyBtn.style.display = 'none';
        downloadZipBtn.style.display = 'none';
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        isProcessing = false;
    }
}
async function checkForUpdates() {
    try {
        const result = await ipcRenderer.invoke('check-updates');
        if (result.success) {
            const d = result.data;
            if (d.hasUpdate) {
                showMessage(`Update available: ${d.latestTag}`, 'success');
            } else {
                showMessage('You are up to date', 'info');
            }
        } else {
            showMessage(`Update check failed: ${result.error}`, 'error');
        }
    } catch (e) {
        showMessage(`Update check error: ${e.message}`, 'error');
    }
}

function setupTTS() {
    function cleanTextForTTS(text) {
        if (!text) return '';
        return text
            .replace(/!\[.*?\]\(.*?\)/g, '')
            .replace(/\[([^\]]+)\]\(.*?\)/g, '$1')
            .replace(/^#+\s/gm, '')
            .replace(/(\*\*|__)(.*?)\1/g, '$2')
            .replace(/(\*|_)(.*?)\1/g, '$2')
            .replace(/`{3}[\s\S]*?`{3}/g, '')
            .replace(/`(.+?)`/g, '$1')
            .replace(/^>\s/gm, '')
            .replace(/^-{3,}$/gm, '')
            .replace(/\.\.\./g, '.')
            .replace(/\.\s*\./g, '.')
            .replace(/\s+/g, ' ')
            .trim();
    }

    let isReading = false;

    if (readAloudBtn) {
        console.log('TTS: Listener attaching');
        readAloudBtn.removeEventListener('click', null); // dummy remove

        readAloudBtn.addEventListener('click', async () => {
            // DEBUG: Alert to confirm click is registered
            // alert('Read button clicked!'); 

            if (isReading) {
                // showMessage('Already reading...', 'info');
                console.warn('TTS: Already reading');
                return;
            }

            let textToRead = currentResultText;
            if (!textToRead && resultsContent) {
                textToRead = resultsContent.innerText;
            }

            if (!textToRead) {
                alert('No text found to read! Please perform OCR first.');
                return;
            }

            textToRead = cleanTextForTTS(textToRead);

            if (!textToRead) {
                alert('Text is empty after cleaning.');
                return;
            }

            isReading = true;
            readAloudBtn.style.display = 'none';
            stopReadBtn.style.display = 'inline-block';
            showMessage('Generating audio...', 'info');

            try {
                const ttsEngine = ttsEngineSelect ? ttsEngineSelect.value : 'edge_tts';
                console.log('TTS: Engine:', ttsEngine);

                const response = await fetch(`${getServerUrl()}/tts`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: textToRead,
                        tts_engine: ttsEngine
                    })
                });

                const result = await response.json();
                console.log('TTS Result:', result);

                if (result.status === 'success' && result.audio_url) {
                    showMessage('Playing audio...', 'success');
                    if (ttsAudio) {
                        ttsAudio.src = `${getServerUrl()}${result.audio_url}`;
                        ttsAudio.onended = () => {
                            isReading = false;
                            readAloudBtn.style.display = 'inline-block';
                            stopReadBtn.style.display = 'none';
                            showMessage('Finished reading', 'success');
                        };
                        ttsAudio.onerror = (e) => {
                            console.error('Audio playback failed, trying browser TTS...');
                            useBrowserTTS(textToRead, isReading, readAloudBtn, stopReadBtn);
                        };
                        try {
                            await ttsAudio.play();
                        } catch (playError) {
                            console.error('Play error, trying browser TTS:', playError);
                            useBrowserTTS(textToRead, isReading, readAloudBtn, stopReadBtn);
                        }
                    }
                } else {
                    // Backend TTS failed, use browser Speech API as fallback
                    console.log('Backend TTS failed, using browser Speech API...');
                    showMessage('Using browser TTS...', 'info');
                    useBrowserTTS(textToRead, isReading, readAloudBtn, stopReadBtn);
                }
            } catch (error) {
                console.error('TTS Network Error, using browser TTS:', error);
                showMessage('Using browser TTS...', 'info');
                useBrowserTTS(textToRead, isReading, readAloudBtn, stopReadBtn);
            }

            // Browser TTS fallback function
            function useBrowserTTS(text, isReadingRef, readBtn, stopBtn) {
                console.log('Using browser TTS fallback...');

                if (!('speechSynthesis' in window)) {
                    alert('Your browser does not support text-to-speech.');
                    isReading = false;
                    readAloudBtn.style.display = 'inline-block';
                    stopReadBtn.style.display = 'none';
                    return;
                }

                // Get voices - may need to wait for them to load
                let voices = speechSynthesis.getVoices();
                if (voices.length === 0) {
                    // Voices not loaded yet, try again after a delay
                    setTimeout(() => {
                        voices = speechSynthesis.getVoices();
                        startBrowserSpeech(text, voices);
                    }, 200);
                } else {
                    startBrowserSpeech(text, voices);
                }

                function startBrowserSpeech(textToSpeak, availableVoices) {
                    console.log('Available voices:', availableVoices.length);
                    availableVoices.forEach((v, i) => console.log(`  ${i}: ${v.name} (${v.lang})`));

                    // Cancel any ongoing speech
                    speechSynthesis.cancel();

                    const utterance = new SpeechSynthesisUtterance(text);

                    // Detect language from text content
                    function detectLanguage(text) {
                        if (!text) return 'en';

                        const len = text.length;

                        // Count character types
                        const arabicChars = (text.match(/[\u0600-\u06FF]/g) || []).length;
                        const chineseChars = (text.match(/[\u4e00-\u9fff]/g) || []).length;
                        const japaneseChars = (text.match(/[\u3040-\u30ff]/g) || []).length;
                        const koreanChars = (text.match(/[\uac00-\ud7af]/g) || []).length;
                        const cyrillicChars = (text.match(/[\u0400-\u04FF]/g) || []).length;

                        // French-specific characters and common words
                        const frenchPattern = /[àâäéèêëïîôùûüÿœæç]|(\b(le|la|les|de|du|des|et|en|un|une|est|que|qui|dans|pour|sur|avec|ce|cette|sont|ont|pas|mais|aussi|plus|tout|comme|elle|il|nous|vous|ils|elles|leur|très|bien|fait|peut|être|avoir|faire|voir|dire|tous|aller|venir|prendre|donner|même|autre|grand|petit|nouveau|premier|dernier|jeune|vieux|beau|bon|mauvais|français|france)\b)/gi;
                        const frenchMatches = text.match(frenchPattern) || [];
                        const frenchScore = frenchMatches.length;

                        // German-specific characters and common words
                        const germanPattern = /[äöüßÄÖÜ]|(\b(der|die|das|und|ist|ein|eine|für|mit|auf|nicht|ich|sie|wir|ihr|es|von|zu|den|dem|sich|als|auch|nach|bei|aus|wenn|noch|werden|haben|sein|werden|kann|so|mehr|sehr|nur|dann|aber|über|vor|können|schon|wieder|gegen|unter|zwischen)\b)/gi;
                        const germanMatches = text.match(germanPattern) || [];
                        const germanScore = germanMatches.length;

                        // Spanish-specific characters and common words
                        const spanishPattern = /[áéíóúñü¿¡]|(\b(el|la|los|las|de|del|en|un|una|es|que|y|por|con|para|se|como|más|su|pero|este|esta|son|sus|al|le|lo|me|ya|muy|sin|sobre|todo|también|bien|puede|donde|cuando|hace|tiene|tengo|hay|entre|así|porque|antes|después|cada|desde|hasta|durante|mediante|según|ser|estar|tener|hacer|poder|decir|ir|ver|dar|saber|querer|llegar|pasar|deber|poner|parecer|quedar|creer|hablar|llevar|dejar|seguir|encontrar|llamar|venir|pensar|salir|volver|tomar|conocer|vivir|sentir|tratar|mirar|contar|empezar|esperar|buscar|existir|entrar|trabajar|escribir|perder|producir|ocurrir|entender|pedir|recibir|recordar|terminar|permitir|aparecer|conseguir|comenzar|servir|sacar|necesitar|mantener|resultar|leer|caer|cambiar|presentar|crear|abrir|considerar|oír|acabar|convertir|ganar|formar|traer|partir|morir|aceptar|realizar|suponer|comprender|lograr|explicar|preguntar|tocar|reconocer|estudiar|alcanzar|nacer|dirigir|correr|utilizar|pagar|ayudar|gustar|jugar|escuchar|cumplir|ofrecer|descubrir|levantar|intentar)\b)/gi;
                        const spanishMatches = text.match(spanishPattern) || [];
                        const spanishScore = spanishMatches.length;

                        const threshold = 0.1;

                        if (arabicChars / len > threshold) return 'ar';
                        if (chineseChars / len > threshold) return 'zh';
                        if (japaneseChars / len > threshold) return 'ja';
                        if (koreanChars / len > threshold) return 'ko';
                        if (cyrillicChars / len > threshold) return 'ru';

                        // Check word-based scoring for Latin languages
                        const wordCount = text.split(/\s+/).length;
                        if (frenchScore > wordCount * 0.1) return 'fr';
                        if (germanScore > wordCount * 0.1) return 'de';
                        if (spanishScore > wordCount * 0.1) return 'es';

                        return 'en'; // Default to English
                    }

                    // Get available voices and find best match
                    const voices = speechSynthesis.getVoices();
                    const detectedLang = detectLanguage(text);
                    console.log('Detected language:', detectedLang);

                    // Find voice for detected language
                    const matchingVoice = availableVoices.find(v => v.lang.startsWith(detectedLang));
                    const englishVoice = availableVoices.find(v => v.lang.startsWith('en'));
                    const frenchVoice = availableVoices.find(v => v.lang.startsWith('fr'));
                    const arabicVoice = availableVoices.find(v => v.lang.startsWith('ar'));

                    if (matchingVoice) {
                        utterance.voice = matchingVoice;
                        utterance.lang = detectedLang;
                        console.log('Using voice:', matchingVoice.name, 'for language:', detectedLang);
                        showMessage(`Using ${matchingVoice.name} for ${detectedLang}`, 'success');
                    } else {
                        // No voice for detected language
                        const availableLangs = [...new Set(availableVoices.map(v => v.lang.split('-')[0]))];
                        console.warn(`No voice found for ${detectedLang}. Available languages:`, availableLangs);

                        if (detectedLang === 'ar' && !arabicVoice) {
                            alert(`No Arabic voice installed on your system.\n\nTo add Arabic voice:\n1. Windows: Settings > Time & Language > Speech > Add voices\n2. Or install Arabic language pack\n\nWill read in English instead.`);
                        } else if (detectedLang === 'fr' && !frenchVoice) {
                            alert(`No French voice installed.\n\nWill read in English instead.`);
                        }

                        if (englishVoice) {
                            utterance.voice = englishVoice;
                            utterance.lang = 'en-US';
                            console.log('Fallback to English voice');
                            showMessage('Using English voice (no ' + detectedLang + ' voice available)', 'warning');
                        }
                    }

                    utterance.onend = () => {
                        isReading = false;
                        readAloudBtn.style.display = 'inline-block';
                        stopReadBtn.style.display = 'none';
                        showMessage('Finished reading', 'success');
                    };

                    utterance.onerror = (e) => {
                        console.error('Browser TTS error:', e);
                        isReading = false;
                        readAloudBtn.style.display = 'inline-block';
                        stopReadBtn.style.display = 'none';
                        showMessage('TTS error: ' + e.error, 'error');
                    };

                    speechSynthesis.speak(utterance);
                }
            }
        });
    }

    if (stopReadBtn) {
        stopReadBtn.addEventListener('click', () => {
            if (ttsAudio) {
                ttsAudio.pause();
                ttsAudio.currentTime = 0;
            }
            isReading = false;
            readAloudBtn.style.display = 'inline-block';
            stopReadBtn.style.display = 'none';
        });
    }
}
