import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the Dataset
file_path = '/Users/zhenghaozhang/hw/4740/project/final_unnormalized_data.csv'
dataset = pd.read_csv(file_path)

# Define Features (X) and Target (y)
X = dataset.drop(columns=['GPA'])  # Features
y = dataset['GPA']  # Target variable

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42, n_estimators=100)
rf_regressor.fit(X_train, y_train)

# Make Predictions on the Test Set
y_pred_rf = rf_regressor.predict(X_test)

# Evaluate the Random Forest Model on the Test Set
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Display evaluation metrics
print("Random Forest Regression Evaluation Metrics (Test Set):")
print(f"Mean Absolute Error (MAE): {mae_rf}")
print(f"Mean Squared Error (MSE): {mse_rf}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"R-squared (R^2): {r2_rf}")

# Perform Cross-Validation
cv_scores = cross_val_score(rf_regressor, X, y, cv=10, scoring='r2')  # 10-fold cross-validation
print("\nCross-Validation R^2 Scores:", cv_scores)
print("Mean R^2 Score (Cross-Validation):", np.mean(cv_scores))
