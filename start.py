#!/usr/bin/env python3
"""
DeepSeek OCR Client Launcher
Handles all installation, GPU detection, and startup logic
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Set UTF-8 encoding for console output (fixes Unicode checkmarks on Windows)
USE_UNICODE = True
if sys.platform == "win32":
    try:
        # Try to set console to UTF-8 mode with proper line buffering
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        # If UTF-8 fails, use ASCII-safe characters
        USE_UNICODE = False

# Define symbols that work with or without UTF-8
CHECK = "√" if USE_UNICODE else "+"
CROSS = "✗" if USE_UNICODE else "X"

def print_header():
    """Print the application header."""
    print("=" * 38)
    print("DeepSeek OCR Client")
    print("=" * 38)
    print()

def check_command(command):
    """Check if a command exists in PATH."""
    return shutil.which(command) is not None

def run_command(command, shell=None, check=True):
    """Run a command and return the result."""
    # On Windows, we need shell=True for batch files like npm
    if shell is None:
        shell = sys.platform == "win32"

    try:
        result = subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            print(f"{CROSS} Command failed: {' '.join(command) if isinstance(command, list) else command}")
            print(f"Error: {e.stderr}")
            sys.exit(1)
        return e
    except FileNotFoundError as e:
        print(f"{CROSS} Command not found: {command[0] if isinstance(command, list) else command}")
        print("  Make sure the command is installed and in your PATH")
        sys.exit(1)

def find_compatible_python():
    """Find a compatible Python version (3.10-3.12) on the system.
    
    Returns:
        str: Path to compatible Python executable, or None if not found
    """
    # Try to find Python using the py launcher (Windows) or python3 command (Unix)
    preferred_versions = ['3.10', '3.11', '3.12']
    
    for version in preferred_versions:
        # Try Windows py launcher first
        try:
            result = subprocess.run(
                ['py', f'-{version}', '--version'],
                capture_output=True,
                text=True,
                check=False,
                shell=(sys.platform == "win32")
            )
            if result.returncode == 0:
                # Found it!
                py_exe = subprocess.run(
                    ['py', f'-{version}', '-c', 'import sys; print(sys.executable)'],
                    capture_output=True,
                    text=True,
                    check=True,
                    shell=(sys.platform == "win32")
                ).stdout.strip()
                return (py_exe, version, result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try python3.X command (Unix/Linux)
        try:
            cmd = f'python{version}'
            result = subprocess.run(
                [cmd, '--version'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                py_exe = subprocess.run(
                    [cmd, '-c', 'import sys; print(sys.executable)'],
                    capture_output=True,
                    text=True,
                    check=True
                ).stdout.strip()
                return (py_exe, version, result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    return None

def check_prerequisites():
    """Check if Node.js and Python are installed."""
    print("Checking prerequisites...")

    # Check Node.js
    if not check_command("node"):
        print(f"{CROSS} Node.js is not installed")
        print("Please install Node.js from https://nodejs.org/")
        input("Press Enter to exit...")
        sys.exit(1)
    print(f"{CHECK} Node.js found")

    # Find compatible Python version
    python_info = find_compatible_python()
    
    if python_info is None:
        print(f"{CROSS} No compatible Python version found!")
        print("")
        print("PyTorch requires Python 3.10, 3.11, or 3.12")
        print("Please install one of these versions:")
        print("  - Python 3.10 (recommended for CUDA): https://www.python.org/downloads/release/python-31014/")
        print("  - Python 3.11: https://www.python.org/downloads/release/python-3119/")
        print("  - Python 3.12: https://www.python.org/downloads/release/python-3126/")
        print("")
        print("After installing, run this script again.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    python_exe, version, version_string = python_info
    
    # Report found Python
    if version == '3.10':
        print(f"{CHECK} Python {version_string} found (optimal for CUDA)")
    elif version == '3.11':
        print(f"{CHECK} Python {version_string} found (good for CUDA)")
    else:  # 3.12
        print(f"{CHECK} Python {version_string} found")
        print("  Note: Python 3.10 is recommended for best CUDA compatibility")
    
    return python_exe

def install_node_dependencies():
    """Install Node.js dependencies if needed."""
    if not Path("node_modules").exists():
        print("\nInstalling Node.js dependencies...")
        result = run_command(["npm", "install"])
        if result.returncode == 0:
            print(f"{CHECK} Node.js dependencies installed")
        else:
            print(f"{CROSS} Failed to install Node.js dependencies")
            sys.exit(1)
    else:
        print(f"{CHECK} Node.js dependencies already installed")

def get_gpu_compute_capability():
    """Get GPU compute capability using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            shell=(sys.platform == "win32")
        )
        compute_cap = result.stdout.strip()
        if compute_cap:
            # Also get GPU name for display
            name_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                shell=(sys.platform == "win32")
            )
            gpu_name = name_result.stdout.strip()

            major = int(compute_cap.split('.')[0])
            print(f"{CHECK} NVIDIA GPU detected: {gpu_name}")
            print(f"  Compute Capability: {compute_cap}")
            return major
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass

    print("! No NVIDIA GPU detected")
    return None

def determine_cuda_version(compute_major):
    """Determine which CUDA version to use based on compute capability.
    
    Compute Capability Guide:
    - 3.x: Kepler (GTX 600/700 series) - CUDA 11.8
    - 5.x-6.x: Maxwell/Pascal (GTX 900/1000 series) - CUDA 12.4
    - 7.x: Volta/Turing (GTX 1600/RTX 20 series) - CUDA 12.4
    - 8.x: Ampere (RTX 30 series) - CUDA 12.4 (optimal with flash-attn)
    - 9.x: Hopper/Ada Lovelace (RTX 40/50 series) - CUDA 12.4 (optimal)
    """
    if compute_major is None:
        print("  No CUDA GPU detected - will use CPU mode")
        return "cpu"
    elif compute_major < 5:
        print("  Using CUDA 11.8 for older GPU (compute capability < 5.0)")
        return "cu118"  # CUDA 11.8 for Kepler and older
    elif compute_major >= 8:
        print("  Using CUDA 12.4 for modern GPU (optimal for flash-attention)")
        return "cu124"  # CUDA 12.4 optimal for Ampere+ (supports flash-attn)
    else:
        print("  Using CUDA 12.4 for GPU (compute capability >= 5.0)")
        return "cu124"  # CUDA 12.4 for Maxwell, Pascal, Turing

def setup_python_environment(python_executable):
    """Set up Python virtual environment and install dependencies.
    
    Args:
        python_executable: Path to the Python executable to use for venv
    """
    venv_path = Path("venv")

    # Create virtual environment if it doesn't exist
    if not venv_path.exists():
        print(f"\nCreating Python virtual environment with {python_executable}...")
        run_command([python_executable, "-m", "venv", "venv"])
        print(f"{CHECK} Virtual environment created")
    else:
        print(f"{CHECK} Virtual environment already exists")

    # Determine the pip executable path
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"

    # Check if PyTorch is already installed
    pytorch_check = subprocess.run(
        [str(python_path), "-c", "import torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
        check=False
    )

    if pytorch_check.returncode != 0:
        # PyTorch not installed, detect GPU and install appropriate version
        print("\nDetecting GPU for PyTorch installation...")
        compute_major = get_gpu_compute_capability()
        cuda_version = determine_cuda_version(compute_major)

        print(f"\nInstalling PyTorch...")
        if cuda_version == "cpu":
            print("  Installing CPU-only version...")
            index_url = "https://download.pytorch.org/whl/cpu"
        elif cuda_version == "cu118":
            print("  Installing with CUDA 11.8 support (for older GPUs)...")
            index_url = "https://download.pytorch.org/whl/cu118"
        else:  # cu124
            print("  Installing with CUDA 12.4 support...")
            index_url = "https://download.pytorch.org/whl/cu124"

        # Install PyTorch (with custom temp dir for downloads)
        env_with_temp = os.environ.copy()
        if 'LOCAL_TEMP_DIR' in os.environ:
            temp_dir = os.environ['LOCAL_TEMP_DIR']
            env_with_temp['TMPDIR'] = temp_dir
            env_with_temp['TEMP'] = temp_dir
            env_with_temp['TMP'] = temp_dir

        # Try installing PyTorch with version constraints first
        try:
            subprocess.run([
                str(pip_path), "install",
                "torch>=2.6.0", "torchvision>=0.21.0", "torchaudio>=2.6.0",
                "--index-url", index_url
            ], env=env_with_temp, check=True, shell=(sys.platform == "win32"))
            print(f"{CHECK} PyTorch installed")
        except subprocess.CalledProcessError:
            print("! PyTorch 2.6.0+ not available, trying PyTorch 2.4.0...")
            try:
                # Fallback to PyTorch 2.4.0 (supports Python 3.8-3.12)
                subprocess.run([
                    str(pip_path), "install",
                    "torch>=2.4.0", "torchvision>=0.19.0", "torchaudio>=2.4.0",
                    "--index-url", index_url
                ], env=env_with_temp, check=True, shell=(sys.platform == "win32"))
                print(f"{CHECK} PyTorch 2.4.0 installed (compatible version)")
            except subprocess.CalledProcessError as e:
                print(f"{CROSS} Failed to install PyTorch")
                print(f"Error: {e}")
                print("")
                print("PyTorch installation failed. This usually means:")
                print("  1. Your Python version is not supported (use 3.10-3.12)")
                print("  2. CUDA version mismatch")
                print("  3. Network connectivity issues")
                print("")
                print("Please try manually installing PyTorch:")
                print(f"  {pip_path} install torch torchvision torchaudio --index-url {index_url}")
                input("Press Enter to exit...")
                sys.exit(1)
    else:
        print(f"{CHECK} PyTorch already installed: {pytorch_check.stdout.strip()}")

    # Check if other dependencies are installed
    deps_check = subprocess.run(
        [str(python_path), "-c", "import flask, flask_cors, PIL, transformers"],
        capture_output=True,
        text=True,
        check=False
    )

    if deps_check.returncode != 0:
        # Install other Python dependencies
        print("\nInstalling Python dependencies...")
        requirements_file = Path("requirements.txt")
        if requirements_file.exists():
            # Read requirements and filter out torch packages (already installed)
            with open(requirements_file) as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                requirements = [req for req in requirements if not any(req.startswith(pkg) for pkg in ['torch', 'torchvision', 'torchaudio'])]

            if requirements:
                # Write filtered requirements to temp file
                temp_req = Path(".requirements.tmp")
                with open(temp_req, 'w') as f:
                    f.write('\n'.join(requirements))

                # Install from temp file (with custom temp dir for downloads)
                env_with_temp = os.environ.copy()
                if 'LOCAL_TEMP_DIR' in os.environ:
                    temp_dir = os.environ['LOCAL_TEMP_DIR']
                    env_with_temp['TMPDIR'] = temp_dir
                    env_with_temp['TEMP'] = temp_dir
                    env_with_temp['TMP'] = temp_dir

                subprocess.run(
                    [str(pip_path), "install", "-r", str(temp_req)],
                    env=env_with_temp,
                    check=True,
                    shell=(sys.platform == "win32")
                )
                temp_req.unlink()  # Remove temp file

        print(f"{CHECK} Python dependencies installed")
    else:
        print(f"{CHECK} Python dependencies already installed")

    # Try to install flash-attn as optional optimization (may fail on some systems)
    print("\nChecking for flash-attention optimization...")
    flash_attn_check = subprocess.run(
        [str(python_path), "-c", "import flash_attn"],
        capture_output=True,
        text=True,
        check=False
    )
    
    if flash_attn_check.returncode != 0:
        # flash-attn not installed, check if we should try to install it
        # Get GPU compute capability from earlier detection
        compute_major = get_gpu_compute_capability()
        
        # flash-attn requires Ampere+ (compute 8.0+) and is very difficult to compile
        if compute_major and compute_major >= 8:
            print("  Your GPU supports flash-attn (Ampere+ architecture)")
            print("  flash-attn installation requires Visual Studio Build Tools")
            print("  Skipping flash-attn for now - you can install manually if needed")
            print("  See: https://github.com/Dao-AILab/flash-attention#installation-and-features")
        elif compute_major:
            print(f"  flash-attn skipped (requires Ampere+ GPU, you have compute {compute_major}.x)")
            print("  Your GPU will still benefit from torch.compile and dtype optimizations!")
        else:
            print("  flash-attn skipped (requires CUDA GPU)")
    else:
        print(f"{CHECK} flash-attn already installed")

    # Apply model syntax fix if needed
    print("\nApplying DeepSeek OCR model patches...")
    patch_script = Path(__file__).parent / "patch_model_syntax.py"
    if patch_script.exists():
        try:
            result = subprocess.run(
                [str(python_path), str(patch_script)],
                capture_output=True,
                text=True,
                check=False
            )
            if "SUCCESS" in result.stdout:
                print(f"{CHECK} Model syntax patches applied")
            elif "already be patched" in result.stdout:
                print(f"{CHECK} Model already patched")
            elif "Cache path not found" in result.stdout:
                print("  Model not downloaded yet - patch will be applied on first use")
        except Exception as e:
            print(f"  Warning: Could not apply model patches: {e}")
    else:
        print("  Warning: patch_model_syntax.py not found")

    return python_path

def start_application(python_path):
    """Start the Electron application."""
    print("\nStarting DeepSeek OCR Client...")

    # Set environment variable for the backend to use the venv Python
    env = os.environ.copy()
    env['PYTHON_PATH'] = str(python_path)

    # Run npm start
    try:
        subprocess.run(["npm", "start"], env=env, check=True, shell=(sys.platform == "win32"))
    except subprocess.CalledProcessError:
        print(f"\n{CROSS} Application exited with an error")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{CHECK} Application closed")
        sys.exit(0)
    except FileNotFoundError:
        print(f"\n{CROSS} npm not found. Make sure Node.js is installed and in your PATH")
        sys.exit(1)

def main():
    """Main entry point."""
    print_header()

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Set up local cache directories (but don't set TEMP globally - it breaks npm)
    cache_dir = Path("cache")
    python_temp_dir = cache_dir / "python-temp"
    python_temp_dir.mkdir(parents=True, exist_ok=True)

    # Set pip cache to local directory
    pip_cache_dir = cache_dir / "pip"
    pip_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['PIP_CACHE_DIR'] = str(pip_cache_dir)

    # Store temp dir path for use in pip commands
    os.environ['LOCAL_TEMP_DIR'] = str(python_temp_dir)

    print(f"Using local cache directory: {cache_dir}")

    # Run setup steps
    python_exe = check_prerequisites()
    install_node_dependencies()
    python_path = setup_python_environment(python_exe)
    start_application(python_path)

if __name__ == "__main__":
    main()