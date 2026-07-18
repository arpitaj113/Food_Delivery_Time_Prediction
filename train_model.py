import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

# Load dataset
df = pd.read_csv("Food_Delivery_Times.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Drop unwanted columns
if "order_id" in df.columns:
    df = df.drop(columns=["order_id"])

# Target
target_column = "delivery_time_min"

X = df.drop(columns=[target_column])
y = df[target_column]

# Feature types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []
best_r2 = -1
best_model = None
best_name = ""

for name, model_algo in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model_algo)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Store results
    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 2)
    })

    print(f"\n{name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.2f}")

    # Save best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
        best_name = name
        best_mae = mae
        best_rmse = rmse

# Results table
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison.csv", index=False)

print("\nModel Comparison")
print(results_df)

print("\n" + "="*40)
print(f"Best Model : {best_name}")
print(f"MAE        : {best_mae:.2f}")
print(f"RMSE       : {best_rmse:.2f}")
print(f"R² Score   : {best_r2:.2f}")
print("="*40)

joblib.dump({
    "model": best_model,
    "mae": best_mae,
    "rmse": best_rmse,
    "r2": best_r2,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "best_model_name": best_name
}, "model.pkl")

print(f"{best_name} saved as model.pkl")
 
