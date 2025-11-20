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
            print(f"✗ Command failed: {' '.join(command) if isinstance(command, list) else command}")
            print(f"Error: {e.stderr}")
            sys.exit(1)
        return e
    except FileNotFoundError as e:
        print(f"✗ Command not found: {command[0] if isinstance(command, list) else command}")
        print("  Make sure the command is installed and in your PATH")
        sys.exit(1)

def check_prerequisites():
    """Check if Node.js and Python are installed."""
    print("Checking prerequisites...")

    # Check Node.js
    if not check_command("node"):
        print("✗ Node.js is not installed")
        print("Please install Node.js from https://nodejs.org/")
        input("Press Enter to exit...")
        sys.exit(1)
    print("✓ Node.js found")

    # Check Python version
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 12):
        print(f"✗ Python {version_info.major}.{version_info.minor} is too old")
        print("Please install Python 3.12+ from https://www.python.org/")
        input("Press Enter to exit...")
        sys.exit(1)
    print(f"✓ Python {version_info.major}.{version_info.minor}.{version_info.micro} found")

def install_node_dependencies():
    """Install Node.js dependencies if needed."""
    if not Path("node_modules").exists():
        print("\nInstalling Node.js dependencies...")
        result = run_command(["npm", "install"])
        if result.returncode == 0:
            print("✓ Node.js dependencies installed")
        else:
            print("✗ Failed to install Node.js dependencies")
            sys.exit(1)
    else:
        print("✓ Node.js dependencies already installed")

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
            print(f"✓ NVIDIA GPU detected: {gpu_name}")
            print(f"  Compute Capability: {compute_cap}")
            return major
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass

    print("! No NVIDIA GPU detected")
    return None

def determine_cuda_version(compute_major):
    """Determine which CUDA version to use based on compute capability."""
    if compute_major is None:
        return "cpu"
    elif compute_major < 5:
        return "cu118"  # CUDA 11.8 for Kepler and older
    else:
        return "cu124"  # CUDA 12.4 for Maxwell and newer (including RTX 50 series)

def setup_python_environment():
    """Set up Python virtual environment and install dependencies."""
    venv_path = Path("venv")

    # Create virtual environment if it doesn't exist
    if not venv_path.exists():
        print("\nCreating Python virtual environment...")
        run_command([sys.executable, "-m", "venv", "venv"])
        print("✓ Virtual environment created")
    else:
        print("✓ Virtual environment already exists")

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

        subprocess.run([
            str(pip_path), "install",
            "torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1",
            "--index-url", index_url
        ], env=env_with_temp, check=True, shell=(sys.platform == "win32"))
        print("✓ PyTorch installed")
    else:
        print(f"✓ PyTorch already installed: {pytorch_check.stdout.strip()}")

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

        print("✓ Python dependencies installed")
    else:
        print("✓ Python dependencies already installed")

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
        print("\n✗ Application exited with an error")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✓ Application closed")
        sys.exit(0)
    except FileNotFoundError:
        print("\n✗ npm not found. Make sure Node.js is installed and in your PATH")
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
    check_prerequisites()
    install_node_dependencies()
    python_path = setup_python_environment()
    start_application(python_path)

if __name__ == "__main__":
    main()