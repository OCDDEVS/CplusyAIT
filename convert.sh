#!/bin/bash

# Next-Gen CPU/GPU AI Framework
# Automated 1.58-bit Post-Training Quantization (PTQ) Script

echo "Compiling Framework Converter..."
cargo build --release --bin convert

echo "Launching Interactive PTQ Quantizer..."
./target/release/convert
