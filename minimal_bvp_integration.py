"""
Minimal BVP-Enhanced EEG Classification
========================================

The LEAST obstructive way to use BVP to improve EEG emotion classification:
Use BVP features as POST-PROCESSING rules to adjust EEG predictions.

This approach:
- Doesn't change the EEG model at all
- No retraining needed
- BVP only adjusts confidence/predictions
- Easy to enable/disable

Author: Final Year Project
Date: 2026-02-24
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report

from eeg_config import Config
from eeg_bilstm_model import SimpleBiLSTMClassifier
from bvp_handcrafted_features import BVPHandcraftedFeatures


# ==================================================
# POST-PROCESSING BVP RULES
# ==================================================

class BVPPostProcessor:
    """
    Use BVP features to post-process EEG predictions.
    
    Rules based on emotion-arousal relationship:
    - High HR + Low HRV → Boost high-arousal emotions (Q1, Q2)
    - Low HR + High HRV → Boost low-arousal emotions (Q3, Q4)
    - Negative valence indicators → Adjust Q1/Q2 vs Q3/Q4
    
    This is the LEAST OBSTRUCTIVE approach - just tweaks final predictions.
    """
    
    def __init__(self, adjustment_strength=0.1):
        """
        Args:
            adjustment_strength: How much to adjust (0.0 = no adjustment, 1.0 = full)
        """
        self.adjustment_strength = adjustment_strength
        
        # Emotion quadrant mapping
        # Q1 = Positive + High Arousal (Enthusiasm)
        # Q2 = Negative + High Arousal (Fear)
        # Q3 = Negative + Low Arousal (Sadness)
        # Q4 = Positive + Low Arousal (Neutral/Calm)
        
        # Arousal indicators
        self.high_arousal_classes = [0, 1]  # Q1, Q2
        self.low_arousal_classes = [2, 3]   # Q3, Q4
        
        # Valence indicators
        self.positive_classes = [0, 3]  # Q1, Q4
        self.negative_classes = [1, 2]  # Q2, Q3
    
    def compute_arousal_score(self, bvp_features):
        """
        Compute arousal score from BVP features.
        
        BVP features: [mean, std, skewness, kurtosis, mean_rr, std_rr, rmssd, 
                       mean_peak_amp, std_peak_amp, num_peaks, heart_rate]
        
        High arousal indicators:
        - High heart rate (index 10)
        - Low HRV/RMSSD (index 6)
        - High peak amplitude variability (index 8)
        
        Returns:
            arousal_score: -1 (low arousal) to +1 (high arousal)
        """
        # Extract relevant features
        heart_rate = bvp_features[10]  # BPM
        rmssd = bvp_features[6]         # HRV measure
        
        # Normalize heart rate (typical range: 60-100 bpm)
        hr_normalized = (heart_rate - 70) / 30.0  # -1 to +1 range
        hr_normalized = np.clip(hr_normalized, -1, 1)
        
        # RMSSD: higher = more relaxed (low arousal)
        # Typical range: 0-50ms, higher = lower arousal
        rmssd_normalized = -rmssd / 50.0  # Negate so high RMSSD = low arousal
        rmssd_normalized = np.clip(rmssd_normalized, -1, 1)
        
        # Combine indicators
        arousal_score = 0.6 * hr_normalized + 0.4 * rmssd_normalized
        
        return arousal_score
    
    def adjust_logits(self, eeg_logits, bvp_features):
        """
        Adjust EEG prediction logits based on BVP arousal.
        
        Args:
            eeg_logits: [batch_size, 4] - Raw logits from EEG model
            bvp_features: [batch_size, 11] - BVP handcrafted features
        
        Returns:
            adjusted_logits: [batch_size, 4] - Adjusted logits
        """
        batch_size = eeg_logits.shape[0]
        adjusted_logits = eeg_logits.clone()
        
        for i in range(batch_size):
            # Compute arousal from BVP
            arousal = self.compute_arousal_score(bvp_features[i].cpu().numpy())
            
            # Adjust logits based on arousal
            adjustment = self.adjustment_strength * arousal
            
            if arousal > 0:  # High arousal → boost Q1, Q2
                adjusted_logits[i, self.high_arousal_classes] += adjustment
                adjusted_logits[i, self.low_arousal_classes] -= adjustment * 0.5
            else:  # Low arousal → boost Q3, Q4
                adjusted_logits[i, self.low_arousal_classes] += abs(adjustment)
                adjusted_logits[i, self.high_arousal_classes] -= abs(adjustment) * 0.5
        
        return adjusted_logits


# ==================================================
# EXAMPLE: Minimal Integration
# ==================================================

def predict_with_bvp_adjustment(eeg_model, eeg_input, bvp_input, 
                                use_bvp_adjustment=True, adjustment_strength=0.1):
    """
    Make predictions with optional BVP adjustment.
    
    This is the MINIMAL way to integrate BVP:
    1. Get EEG predictions (unchanged model)
    2. Optionally adjust with BVP features
    
    Args:
        eeg_model: Pre-trained EEG model
        eeg_input: EEG features [batch, channels, features]
        bvp_input: BVP signals [batch, time_steps]
        use_bvp_adjustment: Whether to use BVP (default: True)
        adjustment_strength: How much to adjust (default: 0.1)
    
    Returns:
        predictions: Class predictions [batch]
        logits: Final logits [batch, n_classes]
    """
    # 1. Get EEG predictions (UNCHANGED)
    eeg_model.eval()
    with torch.no_grad():
        eeg_logits = eeg_model(eeg_input)
    
    # 2. Optionally adjust with BVP
    if use_bvp_adjustment:
        # Extract BVP features
        bvp_extractor = BVPHandcraftedFeatures(sampling_rate=64.0)
        bvp_features = bvp_extractor(bvp_input.unsqueeze(-1))  # [batch, 11]
        
        # Adjust logits
        post_processor = BVPPostProcessor(adjustment_strength=adjustment_strength)
        adjusted_logits = post_processor.adjust_logits(eeg_logits, bvp_features)
        
        # Get predictions from adjusted logits
        predictions = adjusted_logits.argmax(dim=1)
        return predictions, adjusted_logits
    else:
        # Just use EEG predictions
        predictions = eeg_logits.argmax(dim=1)
        return predictions, eeg_logits


# ==================================================
# DEMONSTRATION
# ==================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MINIMAL BVP INTEGRATION DEMO")
    print("=" * 80)
    
    print("\n📋 Approach: Post-Processing BVP Adjustment")
    print("   ✅ No model changes")
    print("   ✅ No retraining needed")
    print("   ✅ Easy to enable/disable")
    print("   ✅ Interpretable rules")
    
    # Create dummy data
    batch_size = 8
    eeg_input = torch.randn(batch_size, 4, 26)  # [batch, channels, features]
    bvp_input = torch.randn(batch_size, 640)    # [batch, time_steps]
    
    # Create dummy EEG model
    eeg_model = SimpleBiLSTMClassifier(
        dx=26, n_channels=4, hidden=256, layers=3, 
        n_classes=4, p_drop=0.4
    )
    eeg_model.eval()
    
    print("\n" + "="*80)
    print("COMPARISON: EEG-only vs EEG+BVP")
    print("="*80)
    
    # Predictions WITHOUT BVP
    preds_eeg_only, logits_eeg = predict_with_bvp_adjustment(
        eeg_model, eeg_input, bvp_input, 
        use_bvp_adjustment=False
    )
    
    # Predictions WITH BVP
    preds_with_bvp, logits_bvp = predict_with_bvp_adjustment(
        eeg_model, eeg_input, bvp_input,
        use_bvp_adjustment=True,
        adjustment_strength=0.2
    )
    
    print("\nSample predictions:")
    print(f"{'Sample':<8} {'EEG-only':<12} {'EEG+BVP':<12} {'Changed?'}")
    print("-" * 50)
    for i in range(batch_size):
        changed = "✓" if preds_eeg_only[i] != preds_with_bvp[i] else ""
        print(f"{i:<8} Q{preds_eeg_only[i].item()+1:<11} Q{preds_with_bvp[i].item()+1:<11} {changed}")
    
    # Show how logits changed
    print("\n" + "="*80)
    print("LOGIT ADJUSTMENTS (Sample 0)")
    print("="*80)
    print(f"{'Class':<8} {'EEG Logit':<12} {'BVP Logit':<12} {'Δ':<10}")
    print("-" * 50)
    for c in range(4):
        delta = logits_bvp[0, c] - logits_eeg[0, c]
        print(f"Q{c+1:<7} {logits_eeg[0, c].item():<12.3f} "
              f"{logits_bvp[0, c].item():<12.3f} {delta.item():+.3f}")
    
    print("\n" + "="*80)
    print("✅ MINIMAL BVP INTEGRATION COMPLETE!")
    print("="*80)
    print("\n💡 How to use:")
    print("   1. Train your EEG model normally (no changes)")
    print("   2. At inference, optionally enable BVP adjustment")
    print("   3. Tune 'adjustment_strength' on validation set")
    print("   4. Compare performance with/without BVP")
    print("\n🎯 Benefits:")
    print("   - Zero model changes")
    print("   - No retraining")
    print("   - Easy A/B testing")
    print("   - Interpretable (arousal-based rules)")
