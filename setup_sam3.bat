@echo off
REM SAM3 Setup Script for Windows
REM Run this in Anaconda Prompt

echo ========================================
echo SAM3 Environment Setup
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Conda not found. Please run this in Anaconda Prompt.
    pause
    exit /b 1
)

echo Step 1: Creating SAM3 environment with Python 3.12...
call conda create -n sam3 python=3.12 -y
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)

echo.
echo Step 2: Activating environment...
call conda activate sam3

echo.
echo Step 3: Installing PyTorch 2.7 with CUDA 12.6...
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Step 4: Cloning SAM3 repository...
cd /d C:\Users\guangwei\Documents\SmartTracker
if exist sam3 (
    echo SAM3 directory exists, pulling latest...
    cd sam3
    git pull
) else (
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3
)

echo.
echo Step 5: Installing SAM3...
pip install -e .
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install SAM3
    pause
    exit /b 1
)

echo.
echo Step 6: Installing additional dependencies...
pip install huggingface_hub opencv-python

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Request access at: https://huggingface.co/facebook/sam3
echo 2. Generate token at: https://huggingface.co/settings/tokens
echo 3. Run: huggingface-cli login
echo 4. Enter your token when prompted
echo.
echo Then you can run:
echo   conda activate sam3
echo   python sam3_annotator.py "D:\Project\Vergin_female_parental\mouse_3_frames\0000" --prompt "mouse"
echo.
pause
