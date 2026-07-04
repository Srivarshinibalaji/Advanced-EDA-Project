import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# STEP 1: LOAD DATASET
# =====================================

df = pd.read_csv("Dataset for Data Analytics - Sheet1.csv")

print("First 5 Rows:")
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())

# =====================================
# STEP 2: HANDLE MISSING VALUES
# =====================================

df["CouponCode"] = df["CouponCode"].fillna(df["CouponCode"].mode()[0])

print("\nMissing Values After Filling:")
print(df.isnull().sum())

# =====================================
# STEP 3: INITIAL BOXPLOT
# =====================================

plt.figure(figsize=(10,6))
df.boxplot(column=["Quantity","UnitPrice","ItemsInCart","TotalPrice"])
plt.title("Box Plot Before Removing Outliers")
plt.savefig("boxplot_before_outliers.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================
# STEP 4: OUTLIER DETECTION
# =====================================

numerical_columns = [
    "Quantity",
    "UnitPrice",
    "ItemsInCart",
    "TotalPrice"
]

print("\nOutlier Count:")

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    print(f"{column}: {len(outliers)} outliers")

# =====================================
# STEP 5: REMOVE OUTLIERS
# =====================================

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[
        (df[column] >= lower) &
        (df[column] <= upper)
    ]

print("\nShape After Removing Outliers:")
print(df.shape)

# =====================================
# STEP 6: FEATURE ENGINEERING
# =====================================

# Feature 1
df["AverageItemPrice"] = df["TotalPrice"] / df["Quantity"]

# Feature 2
df["OrderValueCategory"] = pd.cut(
    df["TotalPrice"],
    bins=[0,500,1500,float("inf")],
    labels=["Low","Medium","High"]
)

# Feature 3
df["DiscountApplied"] = np.where(
    df["CouponCode"].notna(),
    "Yes",
    "No"
)

print("\nNew Features:")
print(df[
    [
        "AverageItemPrice",
        "OrderValueCategory",
        "DiscountApplied"
    ]
].head())

# =====================================
# STEP 7: SAVE CLEANED DATASET
# =====================================

df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")

# =====================================
# STEP 8: ALL GRAPHS
# =====================================

fig, axs = plt.subplots(2,3, figsize=(18,10))

# Graph 1
df["OrderStatus"].value_counts().plot(
    kind="bar",
    ax=axs[0,0]
)
axs[0,0].set_title("Order Status Distribution")

# Graph 2
df["PaymentMethod"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=axs[0,1]
)
axs[0,1].set_title("Payment Method Distribution")
axs[0,1].set_ylabel("")

# Graph 3
axs[0,2].hist(df["TotalPrice"], bins=20)
axs[0,2].set_title("Total Price Distribution")

# Graph 4
axs[1,0].hist(df["UnitPrice"], bins=20)
axs[1,0].set_title("Unit Price Distribution")

# Graph 5
axs[1,1].hist(df["Quantity"], bins=10)
axs[1,1].set_title("Quantity Distribution")

# Graph 6
df.boxplot(
    column=[
        "Quantity",
        "UnitPrice",
        "ItemsInCart",
        "TotalPrice"
    ],
    ax=axs[1,2]
)
axs[1,2].set_title("Box Plot")

plt.tight_layout()

plt.savefig(
    "all_graphs.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nAll graphs saved successfully!")

# =====================================
# STEP 9: CORRELATION MATRIX
# =====================================

plt.figure(figsize=(8,6))

correlation = df[
    [
        "Quantity",
        "UnitPrice",
        "ItemsInCart",
        "TotalPrice",
        "AverageItemPrice"
    ]
].corr()

plt.imshow(
    correlation,
    cmap="coolwarm",
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nCorrelation Matrix saved successfully!")

print("\n======================================")
print("ADVANCED EDA PROJECT COMPLETED")
print("======================================")
print("Generated Files:")
print("1. cleaned_dataset.csv")
print("2. all_graphs.png")
print("3. correlation_matrix.png")
print("4. boxplot_before_outliers.png")
print("======================================")