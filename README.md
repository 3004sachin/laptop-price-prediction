# Laptop Price Predictor

## Overview

This project focuses on building a **Laptop Price Prediction** model using machine learning algorithms. The primary goal is to predict the price of a laptop based on its specifications such as RAM, storage type, processor, screen size, and brand. By analyzing a dataset of laptop models, this project applies various data preprocessing and machine learning techniques to develop an accurate predictive model.

## Project Structure

The project is structured in the following key steps:

1. **Data Collection**:
   - The dataset contains records of different laptops, including features like RAM, storage type (SSD, HDD, etc.), processor type, screen size, brand, and the price.
   - Data is collected from a publicly available source, potentially a CSV file.

2. **Data Preprocessing and Feature Engineering**:
   - **Data Cleaning**: The dataset is cleaned by removing irrelevant columns, handling missing values, and correcting inconsistencies in data.

   - **Feature Engineering**: Memory specifications (like SSD and HDD) are categorized and numerical values are extracted from the raw data. Categorical data such as brand and processor type are converted into numerical form using encoding techniques.

   - **Exploratory Data Analysis (EDA)**: EDA is performed to visualize relationships between features and the price. Plots such as histograms, bar charts, and correlation matrices help understand how different features impact laptop prices.

3. **Model Training**:
   - Several machine learning models are implemented, including **Linear Regression**, **Random Forest**, and **Decision Trees**.
   - The dataset is split into training and testing sets to ensure that the model generalizes well on unseen data.
   - **Feature Selection**: The most important features are selected to improve model performance and reduce complexity.
  

4. **Model Evaluation**:
   - The performance of each model is evaluated using metrics such as **R-squared**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.
   - Models are compared to select the best-performing algorithm for predicting laptop prices.

## How to Run the Project

To run this project locally, follow these steps:

### Prerequisites:
- Python 3.x installed
- Required Python libraries (can be installed using `requirements.txt` or `pip`):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

### Installation Steps:
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/laptop-price-predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd laptop-price-predictor
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook to execute the model training and evaluation:
   ```bash
   jupyter notebook Laptop-Price-Predictor.ipynb
   ```



