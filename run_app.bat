@echo off
echo ==========================================
echo    COVID-19 Detection System Launcher
echo ==========================================

echo [1/3] Installing/Verifying Dependencies...
pip install -r requirements.txt



echo.
echo [3/3] Launching App...
echo.
echo Opening browser...
python -m streamlit run app.py

pause
