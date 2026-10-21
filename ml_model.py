import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# Step 1: Load dataset
data = pd.read_csv("object.csv")

print("Dataset loaded successfully!")
print("Total records:", len(data))

# Step 2: Convert categorical values into numerical values
category_values = {
    value: index
    for index, value in enumerate(data["category"].unique())
}

location_values = {
    value: index
    for index, value in enumerate(data["location"].unique())
}

data["category_code"] = data["category"].map(category_values)
data["location_code"] = data["location"].map(location_values)

# Step 3: Select features for Machine Learning
X = data[["category_code", "location_code"]]

# Step 4: Apply K-Means clustering
if len(data) < 2:
    raise ValueError("At least 2 records are required for clustering.")

model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

data["pattern_cluster"] = model.fit_predict(X) + 1

# Step 5: Save ML analysis results
data.to_csv("object_analysis.csv", index=False)

print("\nML Pattern Analysis")
print("-------------------")

print(
    data[
        ["object", "category", "location", "pattern_cluster"]
    ].to_string(index=False)
)

print("\nPattern clusters created successfully!")
print("object_analysis.csv created successfully!")

# Step 6: Create object frequency chart
os.makedirs("static", exist_ok=True)

object_counts = data["object"].value_counts()

plt.figure(figsize=(9, 5))
object_counts.plot(kind="bar")

plt.title("Lost & Found Object Frequency")
plt.xlabel("Object")
plt.ylabel("Number of Records")

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("static/object_frequency.png", dpi=150)
plt.close()

print("Object frequency chart created successfully!")

# Step 7: Display cluster summary
print("\nCluster Summary")
print("----------------")

print(data["pattern_cluster"].value_counts().sort_index())

print("\nML analysis completed successfully!")