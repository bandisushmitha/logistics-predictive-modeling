import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned Week 2 dataset
df = pd.read_csv("week2_outputs/cleaned_logistics_dataset.csv")

print("Week 3 dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Basic statistics
print("\nBasic Statistics:")
print(df.describe().round(2))

# Regional performance
regional_analysis = df.groupby("region").agg({
    "delivery_time_hr": "mean",
    "transport_cost": "mean",
    "fuel_cost": "mean",
    "distance_km": "mean",
    "on_time": "mean"
}).reset_index()

regional_analysis["on_time"] *= 100

print("\nRegional Performance:")
print(regional_analysis.round(2))

# Traffic analysis
traffic_analysis = df.groupby("traffic").agg({
    "delivery_time_hr": "mean",
    "transport_cost": "mean",
    "on_time": "mean"
}).reset_index()

traffic_analysis["on_time"] *= 100

print("\nTraffic Performance:")
print(traffic_analysis.round(2))

# Vehicle analysis
vehicle_analysis = df.groupby("vehicle_type").agg({
    "delivery_time_hr": "mean",
    "distance_km": "mean",
    "fuel_cost": "mean",
    "transport_cost": "mean",
    "on_time": "mean"
}).reset_index()

vehicle_analysis["on_time"] *= 100

print("\nVehicle Performance:")
print(vehicle_analysis.round(2))

# Category analysis
category_analysis = df.groupby("category").agg({
    "shipment_volume": "mean",
    "delivery_time_hr": "mean",
    "transport_cost": "mean",
    "on_time": "mean"
}).reset_index()

category_analysis["on_time"] *= 100

print("\nCategory Performance:")
print(category_analysis.round(2))

# Save analysis results
regional_analysis.to_csv(
    "week3_outputs/regional_analysis.csv", index=False
)

traffic_analysis.to_csv(
    "week3_outputs/traffic_analysis.csv", index=False
)

vehicle_analysis.to_csv(
    "week3_outputs/vehicle_analysis.csv", index=False
)

category_analysis.to_csv(
    "week3_outputs/category_analysis.csv", index=False
)

print("\nWeek 3 analysis results saved successfully!")

# Visualization 1: Average Delivery Time by Region

plt.figure(figsize=(8, 5))

sns.barplot(
    data=regional_analysis,
    x="region",
    y="delivery_time_hr"
)

plt.title("Average Delivery Time by Region")
plt.xlabel("Region")
plt.ylabel("Delivery Time (Hours)")

plt.tight_layout()

plt.savefig(
    "week3_outputs/delivery_time_by_region.png"
)

plt.show()

print("\nRegional delivery time visualization saved successfully!")

# Visualization 2: Traffic Impact on Delivery Time

plt.figure(figsize=(8, 5))

sns.barplot(
    data=traffic_analysis,
    x="traffic",
    y="delivery_time_hr"
)

plt.title("Impact of Traffic on Delivery Time")
plt.xlabel("Traffic Level")
plt.ylabel("Average Delivery Time (Hours)")

plt.tight_layout()

plt.savefig(
    "week3_outputs/traffic_vs_delivery_time.png"
)

plt.show()

print("\nTraffic visualization saved successfully!")

# Visualization 3: Vehicle Performance

plt.figure(figsize=(8, 5))

sns.barplot(
    data=vehicle_analysis,
    x="vehicle_type",
    y="delivery_time_hr"
)

plt.title("Average Delivery Time by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Delivery Time (Hours)")

plt.tight_layout()

plt.savefig(
    "week3_outputs/delivery_time_by_vehicle.png"
)

plt.show()

print("\nVehicle performance visualization saved successfully!")

# Visualization 4: Transport Cost by Vehicle Type

plt.figure(figsize=(8, 5))

sns.barplot(
    data=vehicle_analysis,
    x="vehicle_type",
    y="transport_cost"
)

plt.title("Average Transport Cost by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Transport Cost")

plt.tight_layout()

plt.savefig(
    "week3_outputs/transport_cost_by_vehicle.png"
)

plt.show()

print("\nTransport cost visualization saved successfully!")

# Visualization 5: Correlation Heatmap

numeric_columns = [
    "shipment_volume",
    "distance_km",
    "fuel_cost",
    "delivery_time_hr",
    "transport_cost"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Logistics Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "week3_outputs/correlation_heatmap.png"
)

plt.show()

print("\nCorrelation heatmap saved successfully!")

# Visualization 6: Distance vs Delivery Time

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="distance_km",
    y="delivery_time_hr",
    hue="traffic"
)

plt.title("Distance vs Delivery Time by Traffic Level")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (Hours)")

plt.tight_layout()

plt.savefig(
    "week3_outputs/distance_vs_delivery_time.png"
)

plt.show()

print("\nDistance vs delivery time visualization saved successfully!")

# Visualization 7: On-Time Delivery by Region

plt.figure(figsize=(8, 5))

sns.barplot(
    data=regional_analysis,
    x="region",
    y="on_time"
)

plt.title("On-Time Delivery Rate by Region")
plt.xlabel("Region")
plt.ylabel("On-Time Delivery Rate (%)")

plt.tight_layout()

plt.savefig(
    "week3_outputs/on_time_delivery_by_region.png"
)

plt.show()

print("\nOn-time delivery visualization saved successfully!")