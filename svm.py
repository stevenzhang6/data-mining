import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

# Load the Dataset
file_path = '/Users/zhenghaozhang/hw/4740/project/final_normalized_data.csv'  # Update this path if necessary
dataset = pd.read_csv(file_path)

# Define Features (X) and Target (y)
X = dataset.drop(columns=['Stress_Level'])  # Features
y = dataset['Stress_Level']  # Target variable

# Normalize the Features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Train the SVM Classifier on Training Set
svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
svm_classifier.fit(X_train, y_train)

# Predict Stress Levels on Test Set
y_pred = svm_classifier.predict(X_test)

# Evaluate the Model (Confusion Matrix and Classification Report)
class_labels = ['Low', 'Moderate', 'High']
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Perform Cross-Validation
cv_scores = cross_val_score(svm_classifier, X_normalized, y, cv=10, scoring='accuracy')

print("\nCross-Validation Results:")
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy (Cross-Validation):", np.mean(cv_scores))
print("Standard Deviation (Cross-Validation):", np.std(cv_scores))

# Compute AUC and ROC Curve for Multi-Class
# Binarize the labels for multi-class ROC
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_binarized.shape[1]

# Train SVM with One-vs-Rest strategy for multi-class ROC
svm_classifier_ovr = OneVsRestClassifier(SVC(kernel='rbf', probability=True, random_state=42))
svm_classifier_ovr.fit(X_train, y_train)

# Predict probabilities for the test set
y_score = svm_classifier_ovr.decision_function(X_test)

# Compute ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red']
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {class_labels[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Customize the plot
plt.title('ROC Curve for Multi-Class SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()
