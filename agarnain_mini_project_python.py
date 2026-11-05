import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def generate_synthetic_data(num_samples=1000):
    """Generates a synthetic student performance dataset for demonstration."""
    np.random.seed(42)
    study_hours = np.random.uniform(1.0, 10.0, num_samples)
    attendance = np.random.uniform(50.0, 100.0, num_samples)
    previous_scores = np.random.uniform(40.0, 95.0, num_samples)
    sleep_hours = np.random.uniform(4.0, 9.0, num_samples)
    
    # Categorical features
    extracurricular = np.random.choice(['Yes', 'No'], num_samples, p=[0.4, 0.6])
    parental_support = np.random.choice(['Low', 'Medium', 'High'], num_samples, p=[0.2, 0.5, 0.3])
    
    # Target variable: Final Exam Score (influenced by features + noise)
    final_score = (
        (study_hours * 3.5) +
        (attendance * 0.3) +
        (previous_scores * 0.4) +
        (sleep_hours * 1.2) +
        np.where(extracurricular == 'Yes', 3.0, 0.0) +
        np.where(parental_support == 'High', 5.0, np.where(parental_support == 'Low', -5.0, 0.0)) +
        np.random.normal(0, 3, num_samples) # Random noise
    )
    # Clip scores to be between 0 and 100
    final_score = np.clip(final_score, 0, 100)
    
    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Attendance': attendance,
        'Previous_Scores': previous_scores,
        'Sleep_Hours': sleep_hours,
        'Extracurricular': extracurricular,
        'Parental_Support': parental_support,
        'Final_Score': final_score
    })
    return df

def main():
    print("--- 1. Loading and Generating Dataset ---")
    df = generate_synthetic_data(1500)
    print(df.head())
    
    # Define Features (X) and Target (y)
    X = df.drop(columns=['Final_Score'])
    y = df['Final_Score']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- 2. Setting up Preprocessing & Pipeline ---")
    numeric_features = ['Study_Hours', 'Attendance', 'Previous_Scores', 'Sleep_Hours']
    categorical_features = ['Extracurricular', 'Parental_Support']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # Create a pipeline combining preprocessing and Random Forest regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    print("\n--- 3. Training the Model ---")
    model.fit(X_train, y_train)
    
    print("\n--- 4. Evaluating the Model ---")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    print("\n--- 5. Making a Sample Prediction ---")
    # Example new student profile
    new_student = pd.DataFrame({
        'Study_Hours': [6.5],
        'Attendance': [88.0],
        'Previous_Scores': [78.5],
        'Sleep_Hours': [7.0],
        'Extracurricular': ['Yes'],
        'Parental_Support': ['High']
    })
    
    predicted_score = model.predict(new_student)
    print(f"Predicted Final Exam Score: {predicted_score[0]:.2f} / 100")

if __name__ == "__main__":
    main()