@echo off
python predict.py
if %ERRORLEVEL% NEQ 0 (
    echo 예측 실행 중 오류가 발생했습니다.
    pause
    exit /b
)
start result_dmt.html
