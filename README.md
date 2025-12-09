# DATA-SCIENCE-JOB-SALARY-PREDICTION

# EECE5644 - Introduction to Machine Learning & Pattern Recognition
[cite_start]**Author:** Krish Santoki (MS in Robotics, Northeastern University) [cite: 4, 5]

## Project Overview
[cite_start]This project aims to predict the **average annual salary** for data science and analyst roles based on job descriptions and company attributes[cite: 7]. [cite_start]The goal is to bring transparency to salary estimation in the data science job market using machine learning techniques[cite: 200].

[cite_start]The problem is formulated as a **regression task** where the target variable is the average salary (in thousands of dollars)[cite: 15].

## Dataset
[cite_start]The project utilizes the **Glassdoor Data Science Job Postings** dataset, consisting of approximately **750 records**[cite: 15].

## Features Used:
* [cite_start]**Categorical:** Job Title, Industry, Sector, State, Ownership [cite: 16-21].
* [cite_start]**Numerical:** Company Rating, Employer Age [cite: 22-24].
* [cite_start]**Text/Unstructured:** Job Description, Skills (parsed for specific keywords)[cite: 9, 10].

## Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost, CatBoost
* [cite_start]**Dimensionality Reduction:** PCA (Principal Component Analysis) [cite: 145]

## Models Implemented
[cite_start]To ensure robust predictions, a wide range of algorithms were implemented and compared[cite: 184]:

1.  [cite_start]**Linear Regression:** Used as a baseline to establish linear trends[cite: 26].
2.  [cite_start]**Ridge Regression:** Implemented with L2 penalty to handle multicollinearity and reduce overfitting[cite: 34].
3.  [cite_start]**Decision Tree:** To capture non-linear interactions between features[cite: 37].
4.  [cite_start]**Random Forest:** An ensemble method using bagging to reduce variance[cite: 64].
5.  [cite_start]**XGBoost Regressor:** A gradient boosting framework tuned for learning rate and depth[cite: 53].
6.  [cite_start]**CatBoost Regressor:** Utilized for its "Ordered Boosting" capability to handle categorical variables efficiently without extensive preprocessing[cite: 61].
7.  [cite_start]**Neural Network (MLP):** A Multi-Layer Perceptron used to capture complex non-linear patterns[cite: 72].
8.  **Stacking Regressor (Final Model):** A meta-ensemble combining Ridge, XGBoost, MLP, and CatBoost. [cite_start]This model uses a meta-learner to optimally blend predictions[cite: 85].

## Methodology
1.  **Preprocessing:** Handling missing values, parsing salary ranges, and feature engineering (extracting skills from text).
2.  [cite_start]**EDA (Exploratory Data Analysis):** Visualizing distributions of job titles and salaries[cite: 104, 116].
3.  [cite_start]**Dimensionality Reduction:** Applied PCA and K-Means clustering to identify inherent groupings in the job postings[cite: 128, 145].
4.  [cite_start]**Modeling & Tuning:** Performed Grid Search with 3-fold Cross-Validation (CV=3) to optimize hyperparameters[cite: 90].
5.  [cite_start]**Evaluation:** Models were evaluated using **R2 Score, MAE, MSE, and RMSE**[cite: 91].

## Key Results
* **Best Model:** Stacking Regressor (Ensemble).
* [cite_start]**Performance:** Achieved an **R2 Score of 0.85**, significantly outperforming the linear baselines[cite: 198].
* **Key Findings:**
    * [cite_start]Job seniority, specific technical skills, and geographic location (e.g., CA, MA) are the strongest predictors of salary[cite: 199].
    * [cite_start]Tree-based ensembles and Stacking methods handled the small dataset size (~750 records) much better than simple linear models[cite: 198].


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

  * [cite_start]**Expand Dataset:** Collect more diverse data to reduce the risk of overfitting[cite: 193].
  * [cite_start]**Advanced NLP:** Implement TF-IDF, Word2Vec, or BERT embeddings to better understand job descriptions context beyond simple keyword matching[cite: 194].
  * [cite_start]**Deployment:** Build an interactive web app for real-time salary estimation[cite: 195].

## Credits

**Project by:** Krish Santoki
**Course:** EECE5644 - Introduction to Machine Learning and Pattern Recognition
[cite_start]**Instructor:** Prof. Roi Yehoshua [cite: 3]
