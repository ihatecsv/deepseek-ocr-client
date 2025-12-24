# DeepSeek-OCR Client

A real-time Electron-based desktop GUI for [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)

**Unaffiliated with [DeepSeek](https://www.deepseek.com/)**

## Features

- Drag-and-drop image upload
- Real-time OCR processing

<img src="docs/images/document.gif" width="1000">

- Click regions to copy 
- Export results as ZIP with markdown images
- GPU acceleration (CUDA, MPS) or CPU fallback

<img src="docs/images/document2.png" width="1000">

## Requirements

- Windows 10/11, other OS are experimental
- Node.js 18+ ([download](https://nodejs.org/))
- Python 3.12+ ([download](https://www.python.org/))
- NVIDIA GPU with CUDA, Apple Silicon (MPS), or CPU

**Note:** MPS and CPU backends use [@Dogacel's modified model](https://huggingface.co/Dogacel/DeepSeek-OCR-Metal-MPS) instead of the [base model](https://huggingface.co/deepseek-ai/DeepSeek-OCR).

## Quick Start (Windows)

1. **Extract** the [ZIP file](https://github.com/ihatecsv/deepseek-ocr-client/archive/refs/heads/main.zip)
2. **Run** `start-client.bat`
   - First run will automatically install dependencies.
   - Subsequent runs will start quicker.
3. **Load Model** - Click the "Load Model" button in the app, this will download or load the model.
   - If this is the first run, this might take some time.
4. **Drop an image** or click the drop zone to select one.
5. **Run OCR** - Click "Run OCR" to process.

Note: if you have issues processing images but the model loads properly, please close and re-open the app and try with the default resolution for "base" and "size". This is a [known issue](https://github.com/ihatecsv/deepseek-ocr-client/issues/2), if you can help to fix it I would appreciate it!

## Linux/macOS

Please follow Windows instructions but start with `start-client.sh` instead of `start-client.bat`.

## Links

- [Model HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Model Blog Post](https://deepseek.ai/blog/deepseek-ocr-context-compression)
- [Model GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)

## Future goals (PRs welcome!)

- [ ] Code cleanup needed (quickly put together)
- [ ] TypeScript
- [ ] Updater from GitHub releases
- [ ] PDF support
- [ ] Batch processing
- [x] CPU/MPS support (thanks [@Dogacel](https://github.com/Dogacel)!)
- [ ] Web version (so you can run the server on a different machine)
- [ ] Better progress bar algo
- [ ] ???

## License

MIT
 