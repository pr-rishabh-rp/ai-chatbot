@echo off
echo Starting AI Voice Chat with GPU Monitoring...
echo.
echo Opening GPU Monitor in new window...
start "GPU Monitor" cmd /k "conda activate aivoice && python gpu_monitor.py 1"
echo.
echo Starting AI Voice Chat in 3 seconds...
timeout /t 3 /nobreak > nul
echo.
conda activate aivoice
python ai_voicetalk_local.py
