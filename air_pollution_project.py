import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the dataset
# Put airPollution.csv in the same folder as this script,
# or replace the path below with the full path on your computer.
dataset = pd.read_csv("C:/Users/Ravi verma/OneDrive/Desktop/Datasets/airPollution.csv")

print("Dataset Shape:", dataset.shape)

print("\nFirst 5 Rows:")
print(dataset.head())

print("\nColumn Names:")
print(dataset.columns.tolist())

print("\nTotal Missing Values:")
print(dataset.isnull().sum().sum())

print("\nMissing Values per Column:")
print(dataset.isnull().sum())


# Step 2: Clean the data
# Rows with no pollutant reading cannot be used to build the target,
# so they are removed.
dataset = dataset.dropna(subset=["pollutant_avg"]).reset_index(drop=True)
print("\nShape after removing rows with missing pollutant_avg:", dataset.shape)


# Step 3: Create High/Low Pollution Target
# Different pollutants (PM2.5, CO, NH3 ...) are measured on different scales,
# so a reading is labelled "High" if it is above the median reading of
# its own pollutant type, otherwise "Low".
median_by_pollutant = dataset.groupby("pollutant_id")["pollutant_avg"].transform("median")

dataset["Pollution_Level"] = np.where(
    dataset["pollutant_avg"] > median_by_pollutant, "High", "Low"
)

print("\nPollution Level Distribution:")
print(dataset["Pollution_Level"].value_counts())

print("\nPollution Level Percentage:")
print(dataset["Pollution_Level"].value_counts(normalize=True) * 100)


# Step 4: Data Visualization

# Graph 1: High/Low Count
plt.figure(figsize=(6, 4))

sns.countplot(
    data=dataset,
    x="Pollution_Level",
    hue="Pollution_Level",
    legend=False
)

plt.title("Pollution Level Distribution")
plt.xlabel("Pollution Level")
plt.ylabel("Number of Readings")
plt.tight_layout()
plt.show()


# Graph 2: Average Pollutant Value Distribution
plt.figure(figsize=(7, 4))

sns.histplot(
    data=dataset,
    x="pollutant_avg",
    bins=30,
    kde=True
)

plt.title("Distribution of Average Pollutant Values")
plt.xlabel("Pollutant Average")
plt.ylabel("Number of Readings")
plt.tight_layout()
plt.show()


# Graph 3: Pollutant Type vs Average Value
plt.figure(figsize=(8, 4))

sns.boxplot(
    data=dataset,
    x="pollutant_id",
    y="pollutant_avg"
)

plt.title("Pollutant Type vs Average Value")
plt.xlabel("Pollutant")
plt.ylabel("Pollutant Average")
plt.tight_layout()
plt.show()


# Graph 4: Top 10 States by Number of Readings
plt.figure(figsize=(8, 5))

top_states = dataset["state"].value_counts().head(10)
sns.barplot(x=top_states.values, y=top_states.index, hue=top_states.index, legend=False)

plt.title("Top 10 States by Number of Readings")
plt.xlabel("Number of Readings")
plt.ylabel("State")
plt.tight_layout()
plt.show()


# Step 5: Prepare Features and Target

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Remove pollutant value columns (min/max/avg) to avoid data leakage,
# plus columns with no useful information:
#   country      -> same value in every row
#   last_update  -> same value in every row
#   station      -> 500+ unique names, too specific
X = dataset.drop(columns=[
    "pollutant_min", "pollutant_max", "pollutant_avg",
    "country", "last_update", "station",
    "Pollution_Level"
])

# Target variable
y = dataset["Pollution_Level"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())


# Step 6: Train-Test Split and Preprocessing

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    exclude=["int64", "float64"]
).columns

# Numerical data preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical data preprocessing
# sparse_output=False gives a dense array, which GaussianNB requires
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Display results
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)


# Step 7: Train Five Machine Learning Algorithms

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Dictionary to store all five models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        random_state=42
    )
}

# Dictionary to store trained pipelines
trained_models = {}

# Train each model
for name, model in models.items():

    # Combine preprocessing and machine learning model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Train the model using training data
    pipeline.fit(X_train, y_train)

    # Store the trained pipeline
    trained_models[name] = pipeline

    print(name, "trained successfully!")

print("\nAll five machine learning models have been trained.")


# Step 8: Evaluate All Five Machine Learning Models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Store evaluation results
results = []

# Evaluate each trained model
for name, model in trained_models.items():

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics ("High" is the positive class)
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, pos_label="High", zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, pos_label="High", zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, pos_label="High", zero_division=0
    )

    # Store results
    results.append({
        "Algorithm": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

    print("\n" + "=" * 50)
    print("Algorithm:", name)
    print("=" * 50)

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test, y_pred, labels=["Low", "High"]
    ))

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

# Display comparison table
results_df = pd.DataFrame(results)

print("\n\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))
