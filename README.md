
```markdown
# Data Science Salary Prediction

Welcome to the **Data Science Salary Prediction** project. This project demonstrates how to predict salaries for data science-related positions using machine learning techniques.

## Project Overview

This repository contains the implementation of a predictive model for data science salaries, based on various features such as experience level, company size, and job title. The project includes the following key steps:

### 1. Data Preprocessing
- **Loading** the dataset from a CSV file.
- **Dropping** irrelevant columns and handling missing values.
- **Encoding** categorical variables using one-hot encoding.

### 2. Exploratory Data Analysis (EDA)
- Visualizing the distribution of salaries with histograms.
- Applying **Power Transformation (Box-Cox)** to normalize salary values.
- Detecting and removing **outliers** using the Interquartile Range (IQR).
- Binning salaries into categories (Low, Medium, High, etc.).

### 3. Model Training
- **Ridge Regression** model is trained on the preprocessed data.
- Evaluated using **Mean Squared Error (MSE)** and **R-squared** metrics.

### 4. Model Deployment with FastAPI
- The model is exposed as an API using **FastAPI** for easy predictions.
- A POST endpoint `/predict` is implemented to predict salaries based on input features.

### 5. API Testing
- The deployed API is tested locally using the POST request to check the predictions.

## Requirements

To run this project, ensure you have the following dependencies:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn fastapi joblib uvicorn requests
```

## Project Structure

```
- data/           : Directory containing the dataset (`ds_salaries.csv`).
- models/          : Saved machine learning model (`ridge_model.sav`).
- app/             : FastAPI application code.
```

## How to Run the API

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the FastAPI application with:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### Example API Usage

To predict a salary, send a POST request to the `/predict` endpoint with the following JSON payload:

```json
{
  "experience_level_encoded": 3.0,
  "company_size_encoded": 3.0,
  "employment_type_PT": 0,
  "job_title_Data_Engineer": 0,
  "job_title_Data_Manager": 1,
  "job_title_Data_Scientist": 0,
  "job_title_Machine_Learning_Engineer": 0
}
```

The response will be the predicted salary in USD:

```json
{
  "Salary (USD)": 105000.00
}
```

## Evaluation Metrics

The model's performance is evaluated using:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual salaries.
  
- **R-squared (R²)**: Represents the proportion of variance in the target variable explained by the model.

## Challenges & Future Work

- Explore additional machine learning models (e.g., Decision Trees, Random Forest) to improve accuracy.
  
- Consider deploying the model in a production environment using **Docker** or **Kubernetes**.
  
- Implement **cross-validation techniques** for better model evaluation.

## License

This project is licensed under the **MIT License**.
```

