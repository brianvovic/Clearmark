@echo off
cd /d "%~dp0..\apps\api"
if exist .venv311\Scripts\activate.bat (
  echo [ClearMark] Dung .venv311 - LaMa + EasyOCR ^(chat luong cao^)
  call .venv311\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
  echo [ClearMark] CANH BAO: chi co .venv - se chay OpenCV, xoa bi mo. Xem README de tao .venv311.
  call .venv\Scripts\activate.bat
)
uvicorn main:app --host 127.0.0.1 --port 8000
