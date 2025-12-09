# DATA-SCIENCE-JOB-SALARY-PREDICTION

# EECE5644 - Introduction to Machine Learning & Pattern Recognition
**Author:** Krish Santoki (MS in Robotics, Northeastern University)

## Project Overview
This project aims to predict the **average annual salary** for data science and analyst roles based on job descriptions and company attributes. The goal is to bring transparency to salary estimation in the data science job market using machine learning techniques.

The problem is formulated as a **regression task** where the target variable is the average salary (in thousands of dollars).

## Dataset
The project utilizes the **Glassdoor Data Science Job Postings** dataset, consisting of approximately **750 records**.

## Features Used:
**Categorical:** Job Title, Industry, Sector, State, Ownership.
**Numerical:** Company Rating, Employer Age.
**Text/Unstructured:** Job Description, Skills (parsed for specific keywords).

## Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost, CatBoost
* **Dimensionality Reduction:** PCA (Principal Component Analysis)

## Models Implemented
To ensure robust predictions, a wide range of algorithms were implemented and compared:

1.  **Linear Regression:** Used as a baseline to establish linear trends.
2.  **Ridge Regression:** Implemented with L2 penalty to handle multicollinearity and reduce overfitting.
3.  **Decision Tree:** To capture non-linear interactions between features.
4.  **Random Forest:** An ensemble method using bagging to reduce variance.
5.  **XGBoost Regressor:** A gradient boosting framework tuned for learning rate and depth.
6.  **CatBoost Regressor:** Utilized for its "Ordered Boosting" capability to handle categorical variables efficiently without extensive preprocessing.
7.  **Neural Network (MLP):** A Multi-Layer Perceptron used to capture complex non-linear patterns.
8.  **Stacking Regressor (Final Model):** A meta-ensemble combining Ridge, XGBoost, MLP, and CatBoost. This model uses a meta-learner to optimally blend predictions.

## Methodology
1.  **Preprocessing:** Handling missing values, parsing salary ranges, and feature engineering (extracting skills from text).
2.  **EDA (Exploratory Data Analysis):** Visualizing distributions of job titles and salaries.
3.  **Dimensionality Reduction:** Applied PCA and K-Means clustering to identify inherent groupings in the job postings.
4.  **Modeling & Tuning:** Performed Grid Search with 3-fold Cross-Validation (CV=3) to optimize hyperparameters.
5.  **Evaluation:** Models were evaluated using **R2 Score, MAE, MSE, and RMSE**.

## Key Results
* **Best Model:** Stacking Regressor (Ensemble).
* **Performance:** Achieved an **R2 Score of 0.85**, significantly outperforming the linear baselines.
* **Key Findings:**
    * Job seniority, specific technical skills, and geographic location (e.g., CA, MA) are the strongest predictors of salary.
    * Tree-based ensembles and Stacking methods handled the small dataset size (~750 records) much better than simple linear models.


## How to Run the Code

### Prerequisites
Ensure you have Python installed. You can install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost catboost
````

### Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/krishsantoki/salary-prediction-project.git](https://github.com/krishsantoki/salary-prediction-project.git)
    cd salary-prediction-project
    ```
2.  **Download the Data:**
    Ensure the Glassdoor dataset CSV file is placed in the project directory (or update the path in the script).
3.  **Run the Notebook/Script:**
      * If using Jupyter Notebook:
        ```bash
        jupyter notebook "Salary Prediction.ipynb"
        ```
      * Run all cells to execute the preprocessing, training, and evaluation pipeline.

## Future Improvements

  * **Expand Dataset:** Collect more diverse data to reduce the risk of overfitting.
  * **Advanced NLP:** Implement TF-IDF, Word2Vec, or BERT embeddings to better understand job descriptions context beyond simple keyword matching.
  * **Deployment:** Build an interactive web app for real-time salary estimation.

## Credits

**Project by:** Krish Santoki
**Course:** EECE5644 - Introduction to Machine Learning and Pattern Recognition
**Instructor:** Prof. Roi Yehoshua
