import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

# Create model directory if it doesn't exist
os.makedirs('./model_reg', exist_ok=True)

cccc = ['price_per_area','building_age','num_bedrooms','area','id_neighbourhood','floor','parking','elevator','storeHouse','balcony','is_luxury','is_modern','janitor','master_room','pool','security','gym']


# Read the data from CSV file
df = pd.read_csv('data/data_final.csv')
df = df[cccc]
df = df[df['price_per_area'] < df['price_per_area'].quantile(0.99)]
df = df[df['price_per_area'] > df['price_per_area'].quantile(0.01)]

# Display basic info about the data
print("Data shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Separate features (X) and target (y)
cols = ['building_age','num_bedrooms','area','id_neighbourhood','floor','parking','elevator','storeHouse','balcony','is_luxury','is_modern','janitor','master_room','pool','security','gym']
X = df[cols]
y = df['price_per_area']

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Standardize the target variable (optional but recommended)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Create base Random Forest model
rf_base = RandomForestRegressor(random_state=42)

# Create GridSearchCV object
print("\n" + "="*80)
print("Starting GridSearchCV for hyperparameter tuning...")
print("="*80)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='r2',  # Optimize for R² score
    n_jobs=-1,  # Use all available processors
    verbose=1  # Print progress
)

# Fit the grid search (this may take a while)
grid_search.fit(X_train_scaled, y_train_scaled)

# Display best parameters and best score
print("\n" + "="*80)
print("GridSearchCV Results:")
print("="*80)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation R² Score: {grid_search.best_score_:.4f}")

# Get the best model
rf_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred_scaled = rf_model.predict(X_test_scaled)

# Inverse transform predictions back to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Evaluate the model with regression metrics (using original scale)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*80)
print("Model Evaluation Metrics (Test Set):")
print("="*80)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Display feature importance
print("\n" + "="*80)
print("Feature Importance:")
print("="*80)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.to_string(index=False))

# Display top 5 hyperparameter combinations from grid search
print("\n" + "="*80)
print("Top 5 Hyperparameter Combinations:")
print("="*80)
cv_results = pd.DataFrame(grid_search.cv_results_)
top_5 = cv_results[['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 
                      'param_min_samples_leaf', 'param_max_features', 'mean_test_score']].head(5)
top_5.columns = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'CV R² Score']
print(top_5.to_string(index=False))

# Save the best model and scalers
print("\n" + "="*80)
print("Saving Model and Scalers...")
print("="*80)

joblib.dump(rf_model, './model_reg/best_model.pkl')
joblib.dump(scaler, './model_reg/scaler_X.pkl')
joblib.dump(scaler_y, './model_reg/scaler_y.pkl')

print(f"✓ Best model saved to: ./model_reg/best_model.pkl")
print(f"✓ Feature scaler saved to: ./model_reg/scaler_X.pkl")
print(f"✓ Target scaler saved to: ./model_reg/scaler_y.pkl")
