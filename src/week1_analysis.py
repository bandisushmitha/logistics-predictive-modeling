import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the logistics dataset
df = pd.read_csv("data/logistics_dataset.csv")

print("Dataset loaded successfully!")

# Check number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nRegions:")
print(df["region"].value_counts())

print("\nVehicle Types:")
print(df["vehicle_type"].value_counts())

print("\nTraffic Conditions:")
print(df["traffic"].value_counts())

print("\nShipment Categories:")
print(df["category"].value_counts())

average_delivery_time = df["delivery_time_hr"].mean()

print("\nAverage Delivery Time:")
print(round(average_delivery_time, 2), "hours")

average_distance = df["distance_km"].mean()

print("\nAverage Distance:")
print(round(average_distance, 2), "km")

average_transport_cost = df["transport_cost"].mean()

print("\nAverage Transport Cost:")
print(round(average_transport_cost, 2))

average_fuel_cost = df["fuel_cost"].mean()

print("\nAverage Fuel Cost:")
print(round(average_fuel_cost, 2))

on_time_rate = df["on_time"].mean() * 100

print("\nOn-Time Delivery Rate:")
print(round(on_time_rate, 2), "%")

regional_analysis = df.groupby("region").agg({
    "delivery_time_hr": "mean",
    "transport_cost": "mean",
    "fuel_cost": "mean",
    "distance_km": "mean",
    "on_time": "mean"
}).reset_index()

regional_analysis["on_time"] = regional_analysis["on_time"] * 100

print("\nRegional Performance:")
print(regional_analysis.round(2))

traffic_analysis = df.groupby("traffic").agg({
    "delivery_time_hr": "mean",
    "transport_cost": "mean",
    "on_time": "mean"
}).reset_index()

traffic_analysis["on_time"] = traffic_analysis["on_time"] * 100

print("\nTraffic Performance:")
print(traffic_analysis.round(2))

vehicle_analysis = df.groupby("vehicle_type").agg({
    "delivery_time_hr": "mean",
    "distance_km": "mean",
    "fuel_cost": "mean",
    "transport_cost": "mean",
    "on_time": "mean"
}).reset_index()

vehicle_analysis["on_time"] = vehicle_analysis["on_time"] * 100

print("\nVehicle Performance:")
print(vehicle_analysis.round(2))

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="region",
    y="delivery_time_hr"
)

plt.title("Average Delivery Time by Region")
plt.xlabel("Region")
plt.ylabel("Delivery Time (Hours)")

plt.tight_layout()
plt.savefig("outputs/delivery_time_by_region.png")

plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="traffic",
    y="delivery_time_hr"
)

plt.title("Delivery Time Distribution by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Delivery Time (Hours)")

plt.tight_layout()
plt.savefig("outputs/delivery_time_by_traffic.png")

plt.show()

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="distance_km",
    y="delivery_time_hr",
    hue="traffic"
)

plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (Hours)")

plt.tight_layout()
plt.savefig("outputs/distance_vs_delivery_time.png")

plt.show()

numeric_columns = [
    "shipment_volume",
    "distance_km",
    "fuel_cost",
    "delivery_time_hr",
    "transport_cost",
    "on_time"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation.round(2))

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Logistics Feature Correlation Matrix")
plt.tight_layout()

plt.savefig("outputs/correlation_matrix.png")

plt.show()

regional_analysis.to_csv("outputs/regional_analysis.csv", index=False)

traffic_analysis.to_csv("outputs/traffic_analysis.csv", index=False)

vehicle_analysis.to_csv("outputs/vehicle_analysis.csv", index=False)

print("\nAnalysis results saved successfully!")