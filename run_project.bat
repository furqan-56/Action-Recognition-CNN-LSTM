@echo off
echo ============================================
echo    Action Recognition - CNN + LSTM
echo ============================================
echo.

cd /d "%~dp0"

echo Installing dependencies...
cd backend
pip install -r requirements.txt
echo.

echo Starting Flask server...
echo Server will run at: http://127.0.0.1:5000
echo.
echo Open frontend\index.html in browser to use the app
echo Press Ctrl+C to stop the server
echo.

python app.py
pause
