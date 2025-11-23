# src/pipeline_components.py

import kfp.dsl as dsl
from kfp import components
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Use type annotations from kfp.dsl for paths and parameters
from kfp.dsl import InputPath, OutputPath

@dsl.component(
    base_image='python:3.9', # Specify a base Docker image
    packages_to_install=['pandas', 'scikit-learn', 'dvc']
)
def data_preprocessing(
    input_data_path: InputPath('CSV'),
    x_train_path: OutputPath('CSV'),
    x_test_path: OutputPath('CSV'),
    y_train_path: OutputPath('CSV'),
    y_test_path: OutputPath('CSV'),
    test_size: float = 0.2
):
    """
    A component to preprocess the data.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Read data
    df = pd.read_csv(input_data_path)

    # Handle missing values if any
    df = df.dropna()

    # Separate features and target
    X = df.drop('MEDV', axis=1)  # Assuming 'MEDV' is the target column
    y = df['MEDV']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for easier saving
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Save the processed data to the output paths provided by KFP
    X_train_df.to_csv(x_train_path, index=False)
    X_test_df.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['scikit-learn', 'pandas', 'joblib']
)
def model_training(
    x_train_path: InputPath('CSV'),
    y_train_path: InputPath('CSV'),
    model_path: OutputPath('Joblib'),
    n_estimators: int = 100
):
    """
    A component to train a Random Forest model.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    # Load the preprocessed data
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze('columns')  # Convert DataFrame to Series

    # Train the model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_path)

# ... (Similar components for data_extraction and model_evaluation)

# Code to compile components to YAML (can be in a separate script)
if __name__ == "__main__":
    from kfp.compiler import Compiler
    
    Compiler().compile(
        pipeline_func=data_preprocessing,
        package_path='../components/data_preprocessing.yaml'
    )
    Compiler().compile(
        pipeline_func=model_training,
        package_path='../components/model_training.yaml'
    )
    # ... compile other components