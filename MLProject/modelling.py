import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Load dataset
data = pd.read_csv("house_preprocessing.csv")

# Misal targetnya SalePrice
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

# Set experiment
mlflow.set_experiment("House Price Prediction")

# MLflow Project (mlflow run .) sudah handle run otomatis

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)

# Logging ke MLflow
mlflow.log_param("model_type", "LinearRegression")
mlflow.log_metric("rmse", rmse)

mlflow.sklearn.log_model(model, "model")

print("RMSE:", rmse)
