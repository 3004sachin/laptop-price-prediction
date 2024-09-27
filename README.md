
# Laptop Price Predictor

## Overview

The **Laptop Price Predictor** is a machine learning web application designed to predict laptop prices based on user-defined configurations. Users can input specifications such as RAM, storage type (SSD, HDD), processor, screen size, and brand to receive accurate price predictions. The project utilizes data preprocessing, feature engineering, and multiple machine learning techniques, with Streamlit providing an interactive user interface.

## Project Structure

The project is structured into several key steps:

### 1. Data Collection
- The dataset contains records for various laptops, including features like RAM, storage type (SSD, HDD), processor type, screen size, brand, and price.
- Data is sourced from a publicly available dataset, as mentioned in the Git repository.

### 2. Data Preprocessing and Feature Engineering
- **Data Cleaning**: The dataset is cleaned by removing irrelevant columns and addressing inconsistencies.
- **Feature Engineering**:
  - Storage types (SSD, HDD) are categorized, and numerical values are extracted.
  - Categorical data such as brand and processor type are encoded numerically.
  - Additional features, such as **Pixels Per Inch (PPI)**, are computed from screen resolution and size.

- **Exploratory Data Analysis (EDA)**:
  - EDA is conducted to uncover relationships between laptop features and prices, revealing key insights:
    - **Brand Impact**: Certain brands (e.g., Apple, Microsoft) generally have higher price ranges.
    - **RAM & Storage**: Higher RAM and SSD configurations significantly increase laptop prices.
    - **Screen Resolution**: Laptops with higher resolution screens (e.g., 4K) tend to have higher prices.
  - Various visualizations (histograms, bar charts, and correlation matrices) help understand feature impacts on the target variable (price).

- **Web App Deployment**: The project is hosted using Streamlit, allowing for user interaction.

### 3. Model Training
- **Models Used**: Multiple machine learning models were evaluated, including:
  - **Linear Regression**
  - **Random Forest Regressor**
  - **Decision Trees**
  - **Extra Trees Regressor**
  - **Gradient Boosting Regressor**
- The dataset is split into training and testing sets to ensure model generalization.
- **Feature Selection**: Relevant features are selected using correlation analysis to improve model accuracy and reduce complexity.
- **Stacked Model**: The final model employs stacking of multiple models (RandomForestRegressor, GradientBoostingRegressor) with Ridge regression as the meta-model, leading to enhanced performance.

### 4. Model Evaluation
- Each model is evaluated using metrics like **R-squared (R²)** and **Mean Absolute Error (MAE)**.
- **Hyperparameter Tuning**: The best-performing models (RandomForestRegressor and GradientBoostingRegressor) are tuned using **RandomizedSearchCV** for optimization.
- **Stacked Model Performance**:
  - Optimized R² Score: **0.8989**
  - Optimized MAE: **0.1530**

## Project Files Structure

```
├── app.py                          # Main Streamlit app file
├── stacked_model.pkl                # Trained and optimized machine learning model
├── df.pkl                           # Preprocessed dataset used in the app
├── Laptop-Price-Predictor.ipynb     # Jupyter notebook for model training
├── requirements.txt                 # Dependencies required to run the project
└── README.md                        # Project documentation
```

## How to Run the Project Locally

### Prerequisites
- Python 3.x installed
- Required Python libraries (install via `requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - streamlit

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/3004sachin/laptop-price-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd laptop-price-predictor
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the model for predictions:
   - To run the **Jupyter Notebook** for model training and evaluation:
     ```bash
     jupyter notebook Laptop-Price-Predictor.ipynb
     ```
   - To run the **Streamlit app** for user interaction:
     ```bash
     streamlit run app.py
     ```

## Technologies Used
- **Python**: Core language for the project.
- **Pandas & NumPy**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning model development.
- **Streamlit**: Web app development and deployment.
- **Matplotlib & Seaborn**: Data visualization for exploratory data analysis.
- **Pickle**: For saving and loading the trained model.

## Model Files
The trained models and processed datasets are saved in pickle format (`stacked_model.pkl` and `df.pkl`). These files can be loaded to make predictions without retraining the model.

## License
This project is licensed under the MIT License.

