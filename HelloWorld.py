import pandas as pd
import numpy as np

df= pd.read_csv("C:\\Users\\akira\\OneDrive\\Desktop\\ds_salaries.csv");
df.head()
df.columns
# Drop the 'Unnamed: 0' column
df = df.drop(columns=['Unnamed: 0'])
df.isnull().sum()  # Check for missing values
df = df.dropna()  # Drop missing values
df = pd.get_dummies(df, drop_first=True)  # Convert categorical variables to dummy/indicator variables

import seaborn as sns
import matplotlib.pyplot as plt

# Plot salary distribution
sns.histplot(df['salary_in_usd'])
plt.title('Salary Distribution')
plt.xlabel('Salary in USD')
plt.ylabel('Frequency')
plt.show() 
from sklearn.preprocessing import PowerTransformer

# Initialize the PowerTransformer (which includes Box-Cox)
transformer = PowerTransformer(method='box-cox')  # Make sure all values are > 0

# Apply the transformation to salary_in_usd
df['power_transformed_salary'] = transformer.fit_transform(df[['salary_in_usd']])

# Check the new distribution
sns.histplot(df['power_transformed_salary'])
plt.title('Power Transformed Salary Distribution')
plt.xlabel('Power Transformed Salary in USD')
plt.ylabel('Frequency')
plt.show()

# Check for outliers using IQR
Q1 = df['power_transformed_salary'].quantile(0.25)
Q3 = df['power_transformed_salary'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
df_no_outliers = df[(df['power_transformed_salary'] >= lower_bound) & (df['power_transformed_salary'] <= upper_bound)]

# Visualize with box plot
sns.boxplot(df_no_outliers['power_transformed_salary'])
plt.title('Box Plot for Outlier Detection (After Power Transformation)')
plt.show()

# Example: Bin the salary into categories
bins = [0, 50000, 100000, 150000, 200000, np.inf]
labels = ['Low', 'Medium', 'High', 'Very High', 'Extreme']
df['salary_binned'] = pd.cut(df['salary_in_usd'], bins=bins, labels=labels)

from sklearn.model_selection import train_test_split

# Define the features (X) and target (y)
X = df.drop(columns=['salary_in_usd'])
y = df['salary_in_usd']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting splits
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train.dtypes)
print(X_test.dtypes)
print(y_train.dtype)
print(y_test.dtype)


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Encoding categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df_encoded.drop(columns=['salary_in_usd'])
y = df_encoded['salary_in_usd']

# Drop non-numeric columns if needed
X = X.drop(columns=['salary_binned'], errors='ignore')  # Drop 'salary_binned'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Align train and test sets
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Fill missing values if any
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# Initialize Ridge Regression model
ridge_model = Ridge(alpha=1.0)

# Fit the model
ridge_model.fit(X_train, y_train)

# Predict on the test set
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Mean Squared Error: {mse_ridge}")
print(f"R-squared: {r2_ridge}")

# Save the trained model using joblib
import joblib
joblib.dump(ridge_model, 'ridge_model.sav')
print("Model saved to 'ridge_model.sav'")

import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialize FastAPI
app = FastAPI()

# Define the request body format for predictions
class PredictionFeatures(BaseModel):
    experience_level_encoded: float
    company_size_encoded: float
    employment_type_PT: int
    job_title_Data_Engineer: int
    job_title_Data_Manager: int
    job_title_Data_Scientist: int
    job_title_Machine_Learning_Engineer: int

# Global variable to store the loaded model
model = None

# Download the model
def download_model():
    global model
    model = joblib.load('ridge_model.sav')  # Adjust the filename to match your saved model

# Download the model immediately when the script runs
download_model()

# API Root endpoint
@app.get("/")
async def index():
    return {"message": "Welcome to the Data Science Income API. Use the /predict feature to predict your income."}

# Prediction endpoint
@app.post("/predict")
async def predict(features: PredictionFeatures):
    # Create input DataFrame for prediction
    input_data = pd.DataFrame([{
        "experience_level_encoded": features.experience_level_encoded,
        "company_size_encoded": features.company_size_encoded,
        "employment_type_PT": features.employment_type_PT,
        "job_title_Data Engineer": features.job_title_Data_Engineer,
        "job_title_Data Manager": features.job_title_Data_Manager,
        "job_title_Data Scientist": features.job_title_Data_Scientist,
        "job_title_Machine Learning Engineer": features.job_title_Machine_Learning_Engineer
    }])

    # Predict using the loaded model
    prediction = model.predict(input_data)[0]

    return {
        "Salary (USD)": prediction
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

import requests

url = 'http://127.0.0.1:8000/predict'

# Example input data
data = {
    "experience_level_encoded": 3.0,
    "company_size_encoded": 3.0,
    "employment_type_PT": 0,
    "job_title_Data_Engineer": 0,
    "job_title_Data_Manager": 1,
    "job_title_Data_Scientist": 0,
    "job_title_Machine_Learning_Engineer": 0
}

# Make a POST request to the API
response = requests.post(url, json=data)

# Print the response
print(response.json())
#http://127.0.0.1:8000


@app.post("/predict")
async def predict(features: PredictionFeatures):
    # Print the received features
    print(f"Received features: {features}")

    # Create input DataFrame for prediction
    input_data = pd.DataFrame([{
        "experience_level_encoded": features.experience_level_encoded,
        "company_size_encoded": features.company_size_encoded,
        "employment_type_PT": features.employment_type_PT,
        "job_title_Data Engineer": features.job_title_Data_Engineer,
        "job_title_Data Manager": features.job_title_Data_Manager,
        "job_title_Data Scientist": features.job_title_Data_Scientist,
        "job_title_Machine Learning Engineer": features.job_title_Machine_Learning_Engineer
    }])

    # Print the input data DataFrame
    print(f"Input Data: {input_data}")

    # Predict using the loaded model
    prediction = model.predict(input_data)[0]

    return {
        "Salary (USD)": prediction
    }
