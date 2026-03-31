import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Load dataset
data = pd.read_csv("house_preprocessing.csv")

# Misal targetnya SalePrice (sesuaikan kalau beda)
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()

# MLflow tracking
mlflow.set_experiment("House Price Prediction")

with mlflow.start_run():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(model, "model")

    print("RMSE:", rmse)