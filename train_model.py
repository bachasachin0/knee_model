# import os
# from joblib import dump
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split

# # Ensure the 'models' directory exists
# if not os.path.exists('models'):
#     os.makedirs('models')

# # Load and preprocess the dataset
# data = pd.read_csv('dataset/updated_data.csv')

# # Label Encoding for categorical features
# label_encoder = LabelEncoder()
# data['Patient_Type'] = label_encoder.fit_transform(data['Patient_Type'])
# data['Flexion_Category'] = label_encoder.fit_transform(data['Flexion_Category'])

# # Prepare features and targets
# X = data[['Patient_Type', 'Flexion_Angle', 'Flexion_Category']]
# y_recovery = data['Recovery_Time_Estimate (Weeks)']
# y_pain = data['pain_curability_percent']

# # Split the data into training and test sets
# X_train, X_test, y_train_recovery, y_test_recovery = train_test_split(X, y_recovery, test_size=0.2, random_state=42)
# _, _, y_train_pain, y_test_pain = train_test_split(X, y_pain, test_size=0.2, random_state=42)

# # Train Recovery Time model
# recovery_time_model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42))
# recovery_time_model.fit(X_train, y_train_recovery)

# # Train Pain Curability model
# pain_curability_model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42))
# pain_curability_model.fit(X_train, y_train_pain)

# # Evaluate Recovery Time Model
# y_pred_recovery = recovery_time_model.predict(X_test)
# recovery_time_r2 = r2_score(y_test_recovery, y_pred_recovery)
# recovery_time_mae = mean_absolute_error(y_test_recovery, y_pred_recovery)

# # Evaluate Pain Curability Model
# y_pred_pain = pain_curability_model.predict(X_test)
# pain_curability_r2 = r2_score(y_test_pain, y_pred_pain)
# pain_curability_mae = mean_absolute_error(y_test_pain, y_pred_pain)

# # Display the model evaluation metrics
# print("### Model Evaluation Metrics ###")
# print(f"Recovery Time Model - R2: {recovery_time_r2:.4f}, MAE: {recovery_time_mae:.4f}")
# print(f"Pain Curability Model - R2: {pain_curability_r2:.4f}, MAE: {pain_curability_mae:.4f}")

# # Save the models with joblib
# # dump(recovery_time_model, 'models/trained_recovery_time_model.joblib')
# # dump(pain_curability_model, 'models/trained_pain_curability_model.joblib')

# print("Models trained and saved successfully.")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump

# Load your dataset
# Assuming you've already loaded your dataset
df = pd.read_csv(r'dataset\updated_data.csv')

# Separate features and target
X = df[['Patient_Type', 'Flexion_Angle', 'Flexion_Category']]  # Feature columns
y_recovery = df['Recovery_Time_Estimate (Weeks)']  # Target for recovery time
y_pain = df['pain_curability_percent']  # Target for pain curability percentage

# Convert categorical columns to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Patient_Type', 'Flexion_Category'], drop_first=True)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train_recovery, y_test_recovery = train_test_split(X, y_recovery, test_size=0.2, random_state=42)
_, _, y_train_pain, y_test_pain = train_test_split(X, y_pain, test_size=0.2, random_state=42)

# Train the Recovery Time model
recovery_time_model = RandomForestRegressor(n_estimators=100, random_state=42)
recovery_time_model.fit(X_train, y_train_recovery)

# Train the Pain Curability model
pain_curability_model = RandomForestRegressor(n_estimators=100, random_state=42)
pain_curability_model.fit(X_train, y_train_pain)

# Evaluate the models
y_pred_recovery = recovery_time_model.predict(X_test)
recovery_time_r2 = r2_score(y_test_recovery, y_pred_recovery)
recovery_time_mae = mean_absolute_error(y_test_recovery, y_pred_recovery)

y_pred_pain = pain_curability_model.predict(X_test)
pain_curability_r2 = r2_score(y_test_pain, y_pred_pain)
pain_curability_mae = mean_absolute_error(y_test_pain, y_pred_pain)

# Print the model evaluation results
print(f"Recovery Time Model R²: {recovery_time_r2:.4f}, MAE: {recovery_time_mae:.4f}")
print(f"Pain Curability Model R²: {pain_curability_r2:.4f}, MAE: {pain_curability_mae:.4f}")

# Save the models using joblib
dump(recovery_time_model, 'models/trained_recovery_time_model.joblib')
dump(pain_curability_model, 'models/trained_pain_curability_model.joblib')

print("Models trained and saved successfully.")
