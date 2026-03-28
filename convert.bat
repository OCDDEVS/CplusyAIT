@echo off
REM Next-Gen CPU/GPU AI Framework
REM Automated 1.58-bit Post-Training Quantization (PTQ) Script

echo Compiling Framework Converter...
cargo build --release --bin convert

echo Launching Interactive PTQ Quantizer...
.\target\release\convert.exe
