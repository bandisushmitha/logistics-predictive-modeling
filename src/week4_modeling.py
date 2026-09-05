import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# Load the cleaned logistics dataset
df = pd.read_csv("week2_outputs/cleaned_logistics_dataset.csv")

print("Week 4 dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

# Define target variable
target = "delivery_time_hr"

# Define input features
features = [
    "region",
    "vehicle_type",
    "traffic",
    "category",
    "shipment_volume",
    "distance_km",
    "fuel_cost",
    "transport_cost"
]

X = df[features]
y = df[target]

print("\nFeatures selected:")
print(features)

print("\nTarget variable:")
print(target)

# Identify categorical and numerical features
categorical_features = [
    "region",
    "vehicle_type",
    "traffic",
    "category"
]

numerical_features = [
    "shipment_volume",
    "distance_km",
    "fuel_cost",
    "transport_cost"
]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Preprocess categorical features using One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

print("\nPreprocessing pipeline created successfully!")

# Linear Regression Model
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

# Train the Linear Regression model
linear_model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")

# Decision Tree Regression Model
decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(
            random_state=42,
            max_depth=8
        ))
    ]
)

# Train the Decision Tree model
decision_tree_model.fit(X_train, y_train)

print("Decision Tree model trained successfully!")

# Random Forest Regression Model
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Train the Random Forest model
random_forest_model.fit(X_train, y_train)

print("Random Forest model trained successfully!")

# Make predictions
models = {
    "Linear Regression": linear_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

results = []

for name, model in models.items():
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# Create model comparison table
results_df = pd.DataFrame(results)

print("\nModel Performance:")
print(results_df.round(4))

# Save model comparison results
results_df.to_csv(
    "week4_outputs/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")

# Generate predictions using the best overall model
best_model = linear_model

test_predictions = best_model.predict(X_test)

predictions_df = X_test.copy()

predictions_df["actual_delivery_time_hr"] = y_test.values
predictions_df["predicted_delivery_time_hr"] = test_predictions

# Calculate prediction error
predictions_df["prediction_error_hr"] = (
    predictions_df["actual_delivery_time_hr"]
    - predictions_df["predicted_delivery_time_hr"]
)

# Save predictions
predictions_df.to_csv(
    "week4_outputs/delivery_time_predictions.csv",
    index=False
)

print("\nDelivery time predictions saved successfully!")

# Save the trained model
joblib.dump(
    best_model,
    "models/logistics_delivery_time_model.pkl"
)

print("Best model saved successfully!")

# Logistics Optimization Analysis

optimization_df = predictions_df.copy()

# Identify high-risk shipments
optimization_df["risk_level"] = np.where(
    optimization_df["predicted_delivery_time_hr"] > 2.5,
    "High",
    np.where(
        optimization_df["predicted_delivery_time_hr"] > 2.0,
        "Medium",
        "Low"
    )
)

# Count shipments by risk level
risk_summary = optimization_df["risk_level"].value_counts().reset_index()

risk_summary.columns = ["risk_level", "shipment_count"]

print("\nShipment Risk Analysis:")
print(risk_summary)

# Save risk analysis
risk_summary.to_csv(
    "week4_outputs/shipment_risk_analysis.csv",
    index=False
)

print("\nShipment risk analysis saved successfully!")

# Optimization Recommendations

recommendations = {
    "High": "Prioritize high-risk shipments, consider alternate routes, and assign suitable vehicles.",
    "Medium": "Monitor traffic conditions and consider route adjustments to reduce delivery delays.",
    "Low": "Continue normal dispatch operations while maintaining standard monitoring."
}

optimization_df["recommendation"] = optimization_df["risk_level"].map(
    recommendations
)

optimization_df.to_csv(
    "week4_outputs/optimization_recommendations.csv",
    index=False
)

print("\nOptimization recommendations saved successfully!")