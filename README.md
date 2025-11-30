# DeepSeek-OCR Client

A real-time Electron-based desktop GUI for [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)

**Unaffiliated with [DeepSeek](https://www.deepseek.com/)**

## Features

- Drag-and-drop image upload
- Real-time OCR processing
- **NEW: Queue processing** - Process multiple files or entire folders automatically
- **NEW: CUDA optimizations** - Up to 6x faster with GPU acceleration

<img src="docs/images/document.gif" width="1000">

- Click regions to copy 
- Export results as ZIP with markdown images
- GPU acceleration (CUDA) with torch.compile optimization
- **NEW: Auto-save** - Queue results organized in timestamped folders

<img src="docs/images/document2.png" width="1000">

## Requirements

- Windows 10/11, other OS are experimental
- Node.js 18+ ([download](https://nodejs.org/))
- Python 3.10-3.12 ([download](https://www.python.org/)) - Note: PyTorch doesn't support 3.13+ yet
- NVIDIA GPU with CUDA (optional but recommended for 6x speedup)

## Quick Start (Windows)

1. **Install Python 3.10-3.12** if not already installed ([Python 3.10 recommended](https://www.python.org/ftp/python/3.10.14/python-3.10.14-amd64.exe))
   - ⚠️ **Important**: Python 3.13+ is not supported (PyTorch limitation)
   - The launcher will automatically detect compatible Python versions
2. **Extract** the [ZIP file](https://github.com/ihatecsv/deepseek-ocr-client/archive/refs/heads/main.zip)
3. **Run** `start-client.bat`
   - First run will automatically:
     - Create a Python virtual environment
     - Install PyTorch with CUDA support
     - Install all dependencies
   - Subsequent runs will start quicker
4. **Load Model** - Click the "Load Model" button in the app (downloads model on first run)
5. **Drop an image** or click the drop zone to select one
6. **Run OCR** - Click "Run OCR" to process

Note: if you have issues processing images but the model loads properly, please close and re-open the app and try with the default resolution for "base" and "size". This is a [known issue](https://github.com/ihatecsv/deepseek-ocr-client/issues/2), if you can help to fix it I would appreciate it!

## Linux/macOS

**Note:** Linux and macOS have not been tested yet. Use `start-client.sh` instead of `start-client.bat`.

**PRs welcome!** If you test on Linux/macOS and encounter issues, please open a pull request with fixes.

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
- [ ] CPU support?
- [ ] Web version (so you can run the server on a different machine)
- [ ] Better progress bar algo
- [ ] ???

## License

MIT
