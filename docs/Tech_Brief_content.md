# Technical Brief: Optimization Roadmap for the India Speaks TTS Engine

**To:** Voice AI Team  
**From:** AI Engineering  
**Date:** 2024-07-12  
**Subject:** Roadmap for Optimizing the TTS Engine for Mobile Deployment

## 1. Executive Summary

This document outlines a strategic roadmap for transitioning the current proof-of-concept (PoC) Text-to-Speech (TTS) engine to a production-ready model that meets the stringent latency and memory requirements for on-device deployment. Our key targets are **≤ 120ms inference time** on a Snapdragon 855 and a **total model size of ≤ 6 MB**.

The current PoC has successfully validated our data pipeline and a lightweight model architecture. The next steps focus on aggressive optimization to achieve our performance goals.

## 2. Proposed Optimization Phases

We recommend a three-pronged approach to optimization, implemented in the following order:

### Phase 1: Model Architecture Refinement & Duration Prediction

The current model uses a simple mean-pooling mechanism to align text and speech, which is not robust for variable-length text.

*   **Action:** Replace the current decoder with a proper duration predictor and upsampler, as seen in models like FastSpeech. This will allow the model to handle arbitrary text inputs and generate more natural-sounding prosody.
*   **Impact:** Improved audio quality and robustness. This is a prerequisite for a functional TTS system.

### Phase 2: Quantization

Quantization reduces the numerical precision of the model's weights and activations, leading to significant reductions in model size and often faster inference.

*   **Action:** Apply **Post-Training Dynamic Quantization** using PyTorch's `torch.quantization` module. This is a straightforward method that quantizes weights to 8-bit integers (INT8) and dynamically quantizes activations at runtime.
*   **Impact:**
    *   **Model Size:** Expected to reduce model size by approximately 4x.
    *   **Latency:** Can lead to a 2x-3x speedup on mobile CPUs that support quantized operations.
*   **Tooling:** Use `torch.quantization.quantize_dynamic`.

### Phase 3: Pruning

Pruning removes redundant or unimportant weights from the model, creating a "sparse" model that can be smaller and faster.

*   **Action:** Implement **magnitude-based weight pruning**. This technique removes weights with the lowest absolute values. The model would then need to be fine-tuned to recover any lost accuracy.
*   **Impact:**
    *   **Model Size:** Can reduce model size by 50-90% depending on the sparsity level.
    *   **Latency:** Speedups are most significant when combined with hardware that can take advantage of sparsity.
*   **Tooling:** Use `torch.nn.utils.prune`.

## 3. On-Device Inference Strategy

Once the model is optimized, we need to deploy it to a mobile-friendly inference engine.

*   **Action:** Convert the optimized PyTorch model to the **ONNX (Open Neural Network Exchange)** format. The ONNX model can then be run using **ONNX Runtime Mobile**, which is highly optimized for ARM CPUs like the Snapdragon series.
*   **Workflow:**
    1.  Export the pruned and quantized PyTorch model to ONNX using `torch.onnx.export`.
    2.  Integrate the ONNX Runtime Mobile library into the India Speaks Android application.
    3.  Load and run the `.onnx` model file for on-device inference.

## 4. Next Steps

1.  Implement the duration predictor in the model architecture.
2.  Begin with post-training quantization as it offers a good balance of effort and reward.
3.  Benchmark the performance of the quantized model on target hardware.
4.  Explore pruning to further reduce model size if the 6 MB target is not met.
