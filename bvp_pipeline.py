"""
BVP Hybrid Encoder Pipeline - Emotion Recognition
==================================================

This pipeline uses BVP Hybrid Encoder (Deep + Handcrafted Features) 
for emotion recognition with real data.

The hybrid approach combines:
- Deep learned features: Conv1d + BiLSTM (64 features)
- Handcrafted features: Statistical + HRV + Pulse (11 features)
- Total: 75-dimensional hybrid representation

Usage:
    python bvp_pipeline.py --head deep --baseline_reduction
    python bvp_pipeline.py --head simple
    python bvp_pipeline.py --eval_only --checkpoint best_bvp_hybrid.pt

Author: Final Year Project
Date: 2026-02-24
"""
