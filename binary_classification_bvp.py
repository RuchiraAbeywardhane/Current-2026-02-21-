import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
# Adjust the path to your actual dataset location
data_path = r"e:\FInal Year Project\MyCodeSpace\Current(2026-02-21)\dataset.csv"  # Update this path
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Assuming your dataset has emotion labels and BVP features
# Adjust column names based on your actual dataset structure
emotion_column = 'emotion'  # Update this to your actual emotion column name

# Display original emotion distribution
print("\nOriginal emotion distribution:")
print(df[emotion_column].value_counts())

# Create binary labels: 0 for neutral, 1 for emotions
df['binary_label'] = df[emotion_column].apply(lambda x: 0 if x.lower() == 'neutral' else 1)

print("\nBinary label distribution (before balancing):")
print(df['binary_label'].value_counts())

# Separate features and target
# Adjust this to select only BVP-related features
X = df.drop([emotion_column, 'binary_label'], axis=1)
y = df['binary_label']

print("\nFeature shape:", X.shape)
print("Features:", X.columns.tolist())

# Apply RandomUnderSampler to balance the classes
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

print("\nBinary label distribution (after undersampling):")
print(pd.Series(y_resampled).value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest classifier
print("\nTraining Random Forest classifier...")
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test_scaled)
y_pred_proba = rf_classifier.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("\n" + "="*50)
print("BINARY CLASSIFICATION RESULTS")
print("="*50)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Neutral', 'Emotional']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xticklabels(['Neutral', 'Emotional'])
axes[0, 0].set_yticklabels(['Neutral', 'Emotional'])

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend(loc='lower right')
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature Importance (Top 15)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False).head(15)

axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'])
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 15 Feature Importances')
axes[1, 0].invert_yaxis()

# 4. Class Distribution
class_dist = pd.DataFrame({
    'Dataset': ['Original', 'After Undersampling', 'After Undersampling'],
    'Class': ['Neutral', 'Neutral', 'Emotional'],
    'Count': [
        df[df['binary_label'] == 0].shape[0],
        pd.Series(y_resampled).value_counts()[0],
        pd.Series(y_resampled).value_counts()[1]
    ]
})

original_emotional = df[df['binary_label'] == 1].shape[0]
class_dist = pd.DataFrame({
    'Dataset': ['Original\nNeutral', 'Original\nEmotional', 'Undersampled\nNeutral', 'Undersampled\nEmotional'],
    'Count': [
        df[df['binary_label'] == 0].shape[0],
        original_emotional,
        pd.Series(y_resampled).value_counts()[0],
        pd.Series(y_resampled).value_counts()[1]
    ]
})

axes[1, 1].bar(class_dist['Dataset'], class_dist['Count'], color=['skyblue', 'salmon', 'lightgreen', 'lightcoral'])
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Class Distribution: Before and After Undersampling')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('binary_classification_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'binary_classification_results.png'")
plt.show()

# Save the model
import joblib
joblib.dump(rf_classifier, 'neutral_vs_emotional_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel saved as 'neutral_vs_emotional_model.pkl'")
print("Scaler saved as 'scaler.pkl'")
