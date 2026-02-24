"""
Diagnostic Script for Data Split Analysis
==========================================

This script analyzes the train/val/test splits to identify potential issues
causing low test accuracy in subject-independent evaluation.

Author: Final Year Project
Date: 2026-02-24
"""

import numpy as np
from collections import Counter
from eeg_config import Config
from eeg_data_loader import load_eeg_data, extract_eeg_features, create_data_splits


def analyze_splits():
    """Analyze data splits for potential issues."""
    
    print("=" * 80)
    print("DATA SPLIT DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    
    # Load configuration
    config = Config()
    
    # Load data
    print("\n📁 Loading EEG data...")
    eeg_X_raw, eeg_y, eeg_subjects, eeg_label_map = load_eeg_data(config.DATA_ROOT, config)
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    print(f"✅ Total samples: {len(eeg_y)}")
    print(f"✅ Total subjects: {len(set(eeg_subjects))}")
    
    # Create splits
    print("\n📊 Creating splits...")
    split_indices = create_data_splits(eeg_y, eeg_subjects, config)
    
    train_idx = split_indices['train']
    val_idx = split_indices['val']
    test_idx = split_indices['test']
    
    # Analyze subjects
    train_subjects = eeg_subjects[train_idx]
    val_subjects = eeg_subjects[val_idx]
    test_subjects = eeg_subjects[test_idx]
    
    train_subj_set = set(train_subjects)
    val_subj_set = set(val_subjects)
    test_subj_set = set(test_subjects)
    
    print("\n" + "=" * 80)
    print("SUBJECT ANALYSIS")
    print("=" * 80)
    
    print(f"\n📋 Subject Distribution:")
    print(f"   Train: {len(train_subj_set)} subjects - {sorted(train_subj_set)}")
    print(f"   Val:   {len(val_subj_set)} subjects - {sorted(val_subj_set)}")
    print(f"   Test:  {len(test_subj_set)} subjects - {sorted(test_subj_set)}")
    
    # Check for overlap
    train_val_overlap = train_subj_set & val_subj_set
    train_test_overlap = train_subj_set & test_subj_set
    val_test_overlap = val_subj_set & test_subj_set
    
    print(f"\n🔍 Subject Overlap Check:")
    print(f"   Train ∩ Val:  {train_val_overlap if train_val_overlap else '✅ No overlap'}")
    print(f"   Train ∩ Test: {train_test_overlap if train_test_overlap else '✅ No overlap'}")
    print(f"   Val ∩ Test:   {val_test_overlap if val_test_overlap else '✅ No overlap'}")
    
    # Analyze sample distribution per subject
    print("\n" + "=" * 80)
    print("SAMPLES PER SUBJECT")
    print("=" * 80)
    
    for split_name, indices in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        subjects = eeg_subjects[indices]
        labels = eeg_y[indices]
        subject_counts = Counter(subjects)
        
        print(f"\n{split_name} Set:")
        for subj in sorted(subject_counts.keys()):
            subj_mask = subjects == subj
            subj_labels = labels[subj_mask]
            label_dist = Counter(subj_labels)
            print(f"   {subj}: {subject_counts[subj]} samples - {dict(label_dist)}")
    
    # Analyze class distribution
    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    train_y = eeg_y[train_idx]
    val_y = eeg_y[val_idx]
    test_y = eeg_y[test_idx]
    
    id2lab = {v: k for k, v in eeg_label_map.items()}
    
    print(f"\n📊 Class Distribution:")
    for split_name, labels in [('Train', train_y), ('Val', val_y), ('Test', test_y)]:
        class_counts = Counter(labels)
        total = len(labels)
        print(f"\n   {split_name} ({total} samples):")
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            pct = 100 * count / total
            print(f"      {id2lab[class_id]}: {count:4d} ({pct:5.1f}%)")
    
    # Check for class imbalance issues
    print("\n" + "=" * 80)
    print("POTENTIAL ISSUES")
    print("=" * 80)
    
    issues_found = []
    
    # Issue 1: Subject overlap
    if train_test_overlap or val_test_overlap:
        issues_found.append("⚠️  CRITICAL: Subject overlap detected (data leakage)")
    
    # Issue 2: Very small test set
    if len(test_idx) < 200:
        issues_found.append(f"⚠️  Small test set: {len(test_idx)} samples (may not be representative)")
    
    # Issue 3: Class imbalance in test set
    test_class_counts = Counter(test_y)
    min_class = min(test_class_counts.values())
    max_class = max(test_class_counts.values())
    if max_class / min_class > 2:
        issues_found.append(f"⚠️  Severe class imbalance in test set (ratio: {max_class/min_class:.1f}:1)")
    
    # Issue 4: Different class distributions
    train_dist = np.array([Counter(train_y)[i] / len(train_y) for i in range(config.NUM_CLASSES)])
    test_dist = np.array([Counter(test_y)[i] / len(test_y) for i in range(config.NUM_CLASSES)])
    dist_diff = np.abs(train_dist - test_dist).max()
    if dist_diff > 0.15:
        issues_found.append(f"⚠️  Train/test distribution mismatch (max diff: {dist_diff:.2%})")
    
    # Issue 5: Few subjects in test set
    if len(test_subj_set) < 3:
        issues_found.append(f"⚠️  Very few test subjects: {len(test_subj_set)} (high variance)")
    
    if issues_found:
        print("\n🚨 Issues Found:")
        for issue in issues_found:
            print(f"   {issue}")
    else:
        print("\n✅ No major issues detected")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("""
1. **Verify Subject Independence**: Ensure no subjects appear in both train and test
2. **Check Test Set Size**: ~237 samples might be too small for reliable evaluation
3. **Monitor Class Balance**: WeightedRandomSampler helps, but test imbalance matters
4. **Consider Cross-Validation**: Use k-fold subject-independent CV for more robust results
5. **Analyze Per-Subject Performance**: Some subjects might be much harder to classify
    """)
    
    print("=" * 80)


if __name__ == "__main__":
    analyze_splits()
