@echo off
title CycleGAN - Face/Sketch Training (RTX 4060)
echo ============================================
echo  CycleGAN Face ^<-^> Sketch  ^|  Local GPU
echo ============================================
echo.

:: Activate your conda/venv environment here if needed
:: call conda activate cyclegan
:: OR: call .venv\Scripts\activate

:: Check if a checkpoint exists to resume from
if exist "checkpoints\latest.pth" (
    echo [Launcher] Resuming from checkpoints\latest.pth
    set RESUME_FLAG=--resume
) else (
    echo [Launcher] No checkpoint found - starting fresh training
    set RESUME_FLAG=
)

echo [Launcher] Starting training on RTX 4060 with AMP...
echo.

python local_train.py %RESUME_FLAG% %*

echo.
echo [Launcher] Training finished or interrupted.
echo [Launcher] Latest checkpoint: checkpoints\latest.pth
pause