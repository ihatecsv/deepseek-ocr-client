const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const axios = require('axios');

let mainWindow;
let pythonProcess;
const PYTHON_SERVER_PORT = 5000;
const PYTHON_SERVER_URL = `http://127.0.0.1:${PYTHON_SERVER_PORT}`;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    icon: path.join(__dirname, 'assets', 'icon.png')
  });

  mainWindow.loadFile('index.html');

  // Open external links in browser instead of Electron window
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Handle navigation to external URLs
  mainWindow.webContents.on('will-navigate', (event, url) => {
    // Allow navigation to local files and the Python server
    if (url.startsWith('file://') || url.startsWith(PYTHON_SERVER_URL)) {
      return;
    }
    // Open external URLs in browser
    event.preventDefault();
    shell.openExternal(url);
  });

  // Open DevTools in development mode
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

function startPythonServer() {
  return new Promise((resolve, reject) => {
    console.log('Starting Python OCR server...');

    const pythonScript = path.join(__dirname, 'backend', 'ocr_server.py');

    // Check if Python script exists
    if (!fs.existsSync(pythonScript)) {
      reject(new Error(`Python script not found: ${pythonScript}`));
      return;
    }

    // Use venv Python if available, otherwise fall back to system Python
    let pythonExecutable;
    if (process.platform === 'win32') {
      const venvPython = path.join(__dirname, 'venv', 'Scripts', 'python.exe');
      pythonExecutable = fs.existsSync(venvPython) ? venvPython : 'python';
    } else {
      const venvPython = path.join(__dirname, 'venv', 'bin', 'python3');
      pythonExecutable = fs.existsSync(venvPython) ? venvPython : 'python3';
    }

    console.log(`Using Python: ${pythonExecutable}`);

    pythonProcess = spawn(pythonExecutable, [pythonScript]);

    let resolved = false;

    const markAsReady = () => {
      if (!resolved) {
        resolved = true;
        console.log('Python server is ready!');
        resolve();
      }
    };

    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python: ${data.toString()}`);

      // Check if server is ready
      if (data.toString().includes('Running on')) {
        markAsReady();
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      // Flask logs to stderr by default, even for INFO messages
      console.log(`Python: ${data.toString()}`);

      // Flask logs to stderr, so also check here for server ready message
      if (data.toString().includes('Running on')) {
        markAsReady();
      }
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
    });

    // Wait for server to start with retry logic (timeout after 30 seconds)
    const startTime = Date.now();
    const maxWaitTime = 30000; // 30 seconds
    const checkInterval = 1000; // Check every 1 second

    const checkWithRetry = async () => {
      if (resolved) return;

      const elapsed = Date.now() - startTime;
      if (elapsed >= maxWaitTime) {
        if (!resolved) {
          resolved = true;
          reject(new Error('Python server failed to start within timeout'));
        }
        return;
      }

      try {
        await checkServerHealth();
        markAsReady();
      } catch (error) {
        // Retry after interval
        setTimeout(checkWithRetry, checkInterval);
      }
    };

    // Start checking after initial delay
    setTimeout(checkWithRetry, 2000);
  });
}

async function checkServerHealth() {
  try {
    const response = await axios.get(`${PYTHON_SERVER_URL}/health`);
    return response.data.status === 'ok';
  } catch (error) {
    throw error;
  }
}

function stopPythonServer() {
  if (pythonProcess) {
    console.log('Stopping Python server...');
    pythonProcess.kill();
    pythonProcess = null;
  }
}

// IPC Handlers
ipcMain.handle('check-server-status', async () => {
  try {
    const response = await axios.get(`${PYTHON_SERVER_URL}/health`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('load-model', async () => {
  try {
    const response = await axios.post(`${PYTHON_SERVER_URL}/load_model`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-model-info', async () => {
  try {
    const response = await axios.get(`${PYTHON_SERVER_URL}/model_info`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('select-image', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'] }
    ]
  });

  if (!result.canceled && result.filePaths.length > 0) {
    return { success: true, filePath: result.filePaths[0] };
  }
  return { success: false };
});

ipcMain.handle('select-images', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'] }
    ]
  });

  if (!result.canceled && result.filePaths.length > 0) {
    return { success: true, filePaths: result.filePaths };
  }
  return { success: false };
});

ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  });

  if (!result.canceled && result.filePaths.length > 0) {
    return { success: true, folderPath: result.filePaths[0] };
  }
  return { success: false };
});

ipcMain.handle('open-folder', async (event, folderPath) => {
  try {
    shell.openPath(folderPath);
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('perform-ocr', async (event, { imagePath, promptType, baseSize, imageSize, cropMode }) => {
  try {
    const FormData = require('form-data');
    const formData = new FormData();

    // Read image file and append to form data
    const imageBuffer = fs.readFileSync(imagePath);
    formData.append('image', imageBuffer, {
      filename: path.basename(imagePath),
      contentType: 'image/jpeg'
    });

    formData.append('prompt_type', promptType || 'document');
    formData.append('base_size', baseSize || 1024);
    formData.append('image_size', imageSize || 640);
    formData.append('crop_mode', cropMode ? 'true' : 'false');

    const response = await axios.post(`${PYTHON_SERVER_URL}/ocr`, formData, {
      headers: formData.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });

    return { success: true, data: response.data };
  } catch (error) {
    console.error('OCR Error:', error);
    return {
      success: false,
      error: error.response?.data?.message || error.message
    };
  }
});

// Queue operations
ipcMain.handle('add-to-queue', async (event, { filePaths, promptType, baseSize, imageSize, cropMode }) => {
  try {
    const FormData = require('form-data');
    const formData = new FormData();

    // Add all files to form data
    for (const filePath of filePaths) {
      const imageBuffer = fs.readFileSync(filePath);
      formData.append('files', imageBuffer, {
        filename: path.basename(filePath),
        contentType: 'image/jpeg'
      });
    }

    formData.append('prompt_type', promptType || 'document');
    formData.append('base_size', baseSize || 1024);
    formData.append('image_size', imageSize || 640);
    formData.append('crop_mode', cropMode ? 'true' : 'false');

    const response = await axios.post(`${PYTHON_SERVER_URL}/queue/add`, formData, {
      headers: formData.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });

    return { success: true, data: response.data };
  } catch (error) {
    console.error('Add to queue error:', error);
    return {
      success: false,
      error: error.response?.data?.message || error.message
    };
  }
});

ipcMain.handle('get-queue-status', async () => {
  try {
    const response = await axios.get(`${PYTHON_SERVER_URL}/queue/status`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('process-queue', async () => {
  try {
    const response = await axios.post(`${PYTHON_SERVER_URL}/queue/process`, {}, {
      timeout: 0 // No timeout for queue processing
    });
    return { success: true, data: response.data };
  } catch (error) {
    console.error('Process queue error:', error);
    return {
      success: false,
      error: error.response?.data?.message || error.message
    };
  }
});

ipcMain.handle('clear-queue', async () => {
  try {
    const response = await axios.post(`${PYTHON_SERVER_URL}/queue/clear`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('remove-from-queue', async (event, itemId) => {
  try {
    const response = await axios.delete(`${PYTHON_SERVER_URL}/queue/remove/${itemId}`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// App lifecycle
app.whenReady().then(async () => {
  try {
    await startPythonServer();
    createWindow();
  } catch (error) {
    console.error('Failed to start Python server:', error);
    dialog.showErrorBox(
      'Startup Error',
      `Failed to start Python server: ${error.message}\n\nPlease make sure Python 3 is installed and the required dependencies are installed (see README.md)`
    );
    app.quit();
  }

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  stopPythonServer();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  stopPythonServer();
});
