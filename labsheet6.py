import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# COMMON FUNCTION: PREPROCESSING, TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate(
    data,
    target_column,
    categorical_columns,
    model_filename,
    project_name
):
    """Preprocess data, compare models, and save the best model."""

    # Separate input features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = [
        column for column in categorical_columns
        if column in X.columns
    ]

    # Handle missing numerical values and scale features
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Handle missing categorical values and encode categories
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Define regression algorithms
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    model_results = {}
    trained_models = {}

    # Train and evaluate each algorithm
    for model_name, algorithm in models.items():

        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", algorithm)
        ])

        model_pipeline.fit(X_train, y_train)

        predictions = model_pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        model_results[model_name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        }

        trained_models[model_name] = model_pipeline

        # Plot actual and predicted values
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, predictions, alpha=0.7)

        minimum_value = min(
            np.min(y_test),
            np.min(predictions)
        )

        maximum_value = max(
            np.max(y_test),
            np.max(predictions)
        )

        plt.plot(
            [minimum_value, maximum_value],
            [minimum_value, maximum_value],
            linestyle="--"
        )

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(
            f"{project_name}: Actual vs Predicted - {model_name}"
        )
        plt.tight_layout()
        plt.show()

    # Compare algorithms using evaluation metrics
    results_df = pd.DataFrame(model_results).T

    # Select the model with the highest R2 score
    best_model_name = results_df["R2 Score"].idxmax()
    best_model = trained_models[best_model_name]

    # Save the selected trained pipeline
    joblib.dump(best_model, model_filename)

    # Plot model comparison
    results_df[["MAE", "RMSE"]].plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title(f"{project_name}: Model Comparison")
    plt.ylabel("Error")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Return trained model and evaluation details
    return best_model, results_df, X_test, y_test


# ============================================================
# PROJECT 1: STUDENT PERFORMANCE PREDICTION SYSTEM
# ============================================================

def student_performance_project():
    """Predict final student marks using supervised learning."""

    dataset_path = "student_performance.csv"

    # Load the student dataset
    student_data = pd.read_csv(dataset_path)

    # Expected columns:
    # attendance
    # internal_marks
    # assignment_score
    # study_hours
    # previous_semester_marks
    # final_marks

    # Select the target variable
    target_column = "final_marks"

    # Student dataset uses numerical features
    categorical_columns = []

    # Train models, compare performance, and save best model
    student_model, student_metrics, X_test, y_test = (
        train_and_evaluate(
            data=student_data,
            target_column=target_column,
            categorical_columns=categorical_columns,
            model_filename="student_performance_model.pkl",
            project_name="Student Performance Prediction"
        )
    )

    # Save the evaluation metrics
    student_metrics.to_csv(
        "student_model_metrics.csv"
    )

    return student_model, student_metrics


# ============================================================
# PROJECT 2: HOUSE PRICE PREDICTION SYSTEM
# ============================================================

def house_price_project():
    """Predict house prices using supervised learning."""

    dataset_path = "house_prices.csv"

    # Load the house price dataset
    house_data = pd.read_csv(dataset_path)

    # Expected columns:
    # area
    # bedrooms
    # location
    # bathrooms
    # parking
    # property_age
    # price

    # Select the target variable
    target_column = "price"

    # Location is a categorical feature
    categorical_columns = ["location"]

    # Train models, compare performance, and save best model
    house_model, house_metrics, X_test, y_test = (
        train_and_evaluate(
            data=house_data,
            target_column=target_column,
            categorical_columns=categorical_columns,
            model_filename="house_price_model.pkl",
            project_name="House Price Prediction"
        )
    )

    # Save the evaluation metrics
    house_metrics.to_csv(
        "house_price_model_metrics.csv"
    )

    return house_model, house_metrics


# ============================================================
# STUDENT PERFORMANCE: PREDICT FOR NEW STUDENT DATA
# ============================================================

def predict_student_performance(
    model,
    attendance,
    internal_marks,
    assignment_score,
    study_hours,
    previous_semester_marks
):
    """Predict final marks for a new student."""

    new_student = pd.DataFrame([{
        "attendance": attendance,
        "internal_marks": internal_marks,
        "assignment_score": assignment_score,
        "study_hours": study_hours,
        "previous_semester_marks": previous_semester_marks
    }])

    prediction = model.predict(new_student)

    return float(prediction[0])


# ============================================================
# HOUSE PRICE: PREDICT FOR A NEW PROPERTY
# ============================================================

def predict_house_price(
    model,
    area,
    bedrooms,
    location,
    bathrooms,
    parking,
    property_age
):
    """Predict the price of a new property."""

    new_house = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "location": location,
        "bathrooms": bathrooms,
        "parking": parking,
        "property_age": property_age
    }])

    prediction = model.predict(new_house)

    return float(prediction[0])


# ============================================================
# LOAD A SAVED MODEL FOR FUTURE USE
# ============================================================

def load_saved_model(model_filename):
    """Load a previously saved machine learning model."""

    if not os.path.exists(model_filename):
        raise FileNotFoundError(
            f"Model file not found: {model_filename}"
        )

    return joblib.load(model_filename)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # Run either project after providing its CSV dataset.
    #
    # For Project 1:
    # student_model, student_metrics = (
    #     student_performance_project()
    # )
    #
    # For Project 2:
    # house_model, house_metrics = house_price_project()
    #
    # To load a saved model:
    # student_model = load_saved_model(
    #     "student_performance_model.pkl"
    # )
    #
    # house_model = load_saved_model(
    #     "house_price_model.pkl"
    # )

    pass
