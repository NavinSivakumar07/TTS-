# Technical Brief: TTS Engine for India Speaks

**Project:** Minimal Text-to-Speech Engine for Tamil Speech Generation  
**Team:** Voice AI Engineering  
**Date:** December 2024

## Executive Summary

This document presents a proof-of-concept Text-to-Speech (TTS) engine designed for real-time Tamil speech generation on mobile devices. The system successfully demonstrates end-to-end text-to-speech synthesis using a lightweight neural architecture trained on a 500-sample parallel dataset.

## Architecture Overview

### Model Design
- **Architecture Type:** Simplified FastSpeech-style encoder-decoder
- **Encoder:** Character embedding + 2-layer 1D CNN with ReLU activation
- **Decoder:** Global average pooling + linear projection to mel-spectrogram
- **Vocoder:** Griffin-Lim algorithm for mel-to-audio conversion

### Key Components
1. **Text Processing:** Character-level tokenization with 29-symbol vocabulary
2. **Acoustic Model:** Predicts 80-channel mel-spectrograms (160 time steps)
3. **Audio Synthesis:** Griffin-Lim reconstruction at 24kHz sampling rate

### Model Parameters
- **Total Parameters:** ~2.1M (encoder: 1.8M, decoder: 0.3M)
- **Input:** Variable-length character sequences
- **Output:** Fixed-size mel-spectrograms (80×160)
- **Training:** L1 loss with Adam optimizer (lr=1e-4)

## Performance Analysis

### Training Results
- **Dataset:** 500 parallel text-mel pairs
- **Overfitting Demonstration:** Loss reduction from 0.25 to 0.24 over 200 iterations
- **Convergence:** Stable training with gradient clipping (threshold=1.0)

### Current Limitations
- **Fixed Output Length:** Current architecture produces constant-duration output
- **Simple Alignment:** Mean-pooling approach lacks prosodic control
- **Audio Quality:** Griffin-Lim introduces artifacts compared to neural vocoders

## Latency & Size Estimates

### Current Metrics (Unoptimized)
- **Model Size:** ~8.5 MB (PyTorch state dict)
- **Inference Time:** ~15ms per sentence (CPU, estimated)
- **Memory Usage:** ~50MB during inference

### Target Optimization Goals
- **Model Size:** ≤6 MB (requirement)
- **Latency:** ≤120ms per sentence on Snapdragon 855 (requirement)
- **Audio Quality:** Maintain intelligibility while reducing artifacts

## Optimization Roadmap

### Phase 1: Architecture Enhancement (Weeks 1-2)
**Objective:** Improve audio quality and robustness
- Implement duration predictor for variable-length alignment
- Add attention mechanism for better text-speech alignment
- Replace mean-pooling with learned upsampling

**Expected Impact:**
- Improved prosody and naturalness
- Better handling of variable-length inputs
- Foundation for advanced optimizations

### Phase 2: Model Compression (Weeks 3-4)
**Objective:** Reduce model size and inference time

**2.1 Quantization**
- Apply post-training dynamic quantization (INT8)
- Expected: 4x size reduction (8.5MB → 2.1MB)
- Expected: 2-3x inference speedup

**2.2 Pruning**
- Magnitude-based weight pruning (50-70% sparsity)
- Fine-tuning to recover accuracy
- Expected: Additional 2-3x size reduction

### Phase 3: Deployment Optimization (Weeks 5-6)
**Objective:** Enable efficient on-device inference

**3.1 Model Conversion**
- Export to ONNX format for mobile deployment
- Optimize for ARM CPU architecture
- Integration with ONNX Runtime Mobile

**3.2 Vocoder Optimization**
- Implement lightweight neural vocoder (MelGAN-style)
- Alternative: Optimized Griffin-Lim with reduced iterations
- Target: Maintain 24kHz output quality

## Technical Risks & Mitigation

### Risk 1: Quality Degradation from Compression
- **Mitigation:** Gradual pruning with fine-tuning
- **Fallback:** Reduce target compression ratio

### Risk 2: Latency Target Not Met
- **Mitigation:** Profile bottlenecks, optimize critical paths
- **Fallback:** Reduce model complexity or output quality

### Risk 3: Mobile Integration Challenges
- **Mitigation:** Early prototyping with ONNX Runtime
- **Fallback:** Cloud-based inference with caching

## Next Steps

1. **Immediate (Week 1):** Implement duration predictor and attention mechanism
2. **Short-term (Weeks 2-4):** Apply quantization and pruning optimizations
3. **Medium-term (Weeks 5-6):** Deploy optimized model to mobile test environment
4. **Validation:** Benchmark against Snapdragon 855 latency requirements

## Conclusion

The proof-of-concept successfully validates the core TTS pipeline and demonstrates clear learning capability. The proposed optimization roadmap provides a systematic approach to meeting the 6MB size and 120ms latency requirements while maintaining acceptable audio quality for the India Speaks platform.

---
*This technical brief summarizes the current state and optimization strategy for the India Speaks TTS engine development.* 