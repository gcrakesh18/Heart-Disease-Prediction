import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Create plots directory
plots_dir = 'app/static/plots'
os.makedirs(plots_dir, exist_ok=True)

print(f"Generating test plots in {plots_dir}")

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = np.array([[120, 30], [20, 140]])  # Example confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'Heart Disease'], 
            yticklabels=['Healthy', 'Heart Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{plots_dir}/confusion_matrix.png')
plt.close()
print(f"Generated confusion matrix")

# 2. ROC Curve
plt.figure(figsize=(8, 6))
fpr = np.linspace(0, 1, 100)
tpr = np.sqrt(fpr)  # Example curve shape better than random
plt.plot(fpr, tpr, label=f'AUC = 0.94')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig(f'{plots_dir}/roc_curve.png')
plt.close()
print(f"Generated ROC curve")

# 3. Feature Importance
plt.figure(figsize=(12, 8))
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']
importance = np.array([0.18, 0.15, 0.14, 0.12, 0.10, 0.09, 0.08, 0.06, 0.05, 0.03])  # Example values
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
sns.barplot(data=feature_importance_df, x='Importance', y='Feature')
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig(f'{plots_dir}/feature_importance.png')
plt.close()
print(f"Generated feature importance plot")

# 4. Learning Curve
plt.figure(figsize=(10, 6))
# Example data for learning curve
train_sizes = np.linspace(0.1, 1.0, 10)
train_mean = 0.75 + 0.15 * np.log(train_sizes)
train_std = 0.05 - 0.02 * train_sizes
test_mean = 0.70 + 0.10 * np.log(train_sizes)
test_std = 0.08 - 0.03 * train_sizes

plt.plot(train_sizes, train_mean, label='Training accuracy', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, label='Validation accuracy', color='green', marker='s')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')

plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.grid(True)
plt.savefig(f'{plots_dir}/learning_curve.png')
plt.close()
print(f"Generated learning curve")

# 5. Precision-Recall Curve
plt.figure(figsize=(8, 6))
recall = np.linspace(0, 1, 100)
precision = 1 - (0.5 * recall)  # Example precision-recall curve
plt.plot(recall, precision, label=f'AUC = 0.87')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.savefig(f'{plots_dir}/precision_recall_curve.png')
plt.close()
print(f"Generated precision-recall curve")

print("All test plots generated successfully") 