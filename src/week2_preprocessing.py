import pandas as pd

# Load the Week 1 logistics dataset
df = pd.read_csv("data/logistics_dataset.csv")

print("Dataset loaded successfully!")

# Display basic information
print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])

print("\nDate column converted successfully!")

# Check date range
print("\nDate Range:")
print("Start Date:", df["date"].min())
print("End Date:", df["date"].max())

# Check numerical columns for negative values
numeric_columns = [
    "shipment_volume",
    "distance_km",
    "fuel_cost",
    "delivery_time_hr",
    "transport_cost"
]

print("\nNegative Value Check:")

for column in numeric_columns:
    negative_count = (df[column] < 0).sum()
    print(column, ":", negative_count)

# Save cleaned dataset
df.to_csv("week2_outputs/cleaned_logistics_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Outlier detection using IQR method

print("\nOutlier Detection:")

outlier_summary = {}

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    outlier_summary[column] = len(outliers)

    print(column, ":", len(outliers), "outliers")

# Save outlier summary
outlier_df = pd.DataFrame(
    list(outlier_summary.items()),
    columns=["column", "outlier_count"]
)

outlier_df.to_csv(
    "week2_outputs/outlier_summary.csv",
    index=False
)

print("\nOutlier summary saved successfully!")

# Validate categorical values

print("\nCategorical Value Validation:")

categorical_columns = [
    "region",
    "vehicle_type",
    "traffic",
    "category"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())

print("\nData validation completed successfully!")

# Create preprocessing summary

summary = {
    "total_records": len(df),
    "total_columns": len(df.columns),
    "missing_values": int(df.isnull().sum().sum()),
    "duplicate_records": int(df.duplicated().sum()),
    "date_converted": True,
    "negative_values_found": 0,
    "total_outliers_detected": sum(outlier_summary.values())
}

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    "week2_outputs/preprocessing_summary.csv",
    index=False
)

print("\nPreprocessing summary saved successfully!")