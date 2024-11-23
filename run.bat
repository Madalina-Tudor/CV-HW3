@echo off
:: Script Name: run.bat
:: Purpose: Automate running the part2.py script with environment setup and dependency installation.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not added to PATH. Please install it and try again.
    pause
    exit /b
)

:: Ensure the virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo Activating the virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Setting up a new one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing required dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install required dependencies. Check your internet connection or the requirements.txt file.
        pause
        exit /b
    )
)

:: Run the Python script
echo Running the part2.py script...
python part2.py
if errorlevel 1 (
    echo ERROR: An error occurred while running the script. Check the script for errors.
    pause
    exit /b
)

echo Script executed successfully! Results are saved in the specified output folder.
pause
