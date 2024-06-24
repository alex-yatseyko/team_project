# Project: Credit Risk Prediction
## 1. Folder Structure

|-- project

|---- notebooks

|------ data_preparation.ipynb

|------ exploratory_analysis.ipynb

|------ model_training.ipynb

|---- saves 

|---- tests

|---- data

|------ processed

|-------- processed_credit_risk_dataset.csv

|------ raw

|-------- credit_risk_dataset.csv

|------ sql

|-------- processed_credit_risk_dataset.db

|-- reports

|-- src

|-- README.md

|-- README_dsi.md

|-- .gitignore


- Data: Contains the raw, processed and sql data.

- Project: A folder for project files.

- Notebooks: A folder with notebooks we used: data_preparation.ipynb, exploratory_analysis.ipynb, model_training.ipynb.

- Reports: Generated HTML, PDF etc.

- src: Project source code.

- README: This file.

- README_dsi.md: Old README file.

- .gitignore: Files to exclude from this folder, specified by the Technical Facilitator


## 2. Project Definition and Objective

**Objective:** Develop a machine learning model to predict the likelihood of a loan applicant defaulting.

**Goal:** Create a robust model to help financial institutions assess the risk associated with granting loans.



## 3. Rules of Engagement 

- Treat everyone with respect

- Assume positive intent, and then assume nothing else

- State views and ask genuine questions

- Celebrate each other's accomplishments

- Make mistakes


## 4. Main Question

**What factors contribute to the likelihood of a borrower defaulting on a loan?**

### Supporting Questions

#### Demographic Factors
- How do age and income levels influence the probability of loan default?

#### Employment and Home Ownership
- Does the length of employment or type of home ownership (e.g., rent vs. own) correlate with loan default rates?

#### Loan Characteristics
- How do different loan characteristics such as loan amount, interest rate, and loan grade affect the likelihood of default?

#### Credit History
- What is the impact of credit history length and the presence of defaults on file on the loan default probability?

#### Loan Intent
- Are certain loan intents (e.g., debt consolidation, home improvement) more associated with higher default rates?

### Objectives

#### Identify Key Factors
- Determine the key features that are most indicative of a borrower’s likelihood to default on a loan.

#### Feature Relationships
- Analyze relationships between different features and their impact on loan default probability.

#### Predictive Modeling
- Build a predictive model that can accurately predict whether a borrower will default on a loan based on their profile and loan characteristics.

#### Insights for Financial Institutions
- Provide actionable insights for financial institutions to improve their credit risk assessment processes.

## 5. Data Collection

**Source:** [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

Dataset was chosen based on strict criteria:

- Relevance: Dataset included various different features which can affect the loan status.

- Completeness: Carefully filled out missing values in data.

- Format: Preferred CSV for compatibility. 

- Preprocessing: Used 1-hot encoding for categorical data. 

- Balance: Applied stratify paramiter for balanced training and testing sets. 

- Algorithms: random forest, decision tree, logistic regression with increased iterations and KNN. 

- Evaluation: Assessed with  Precision, Recall, and F1 score due to data imbalance.

| Feature Name                 | Description                      |
|------------------------------|----------------------------------|
| `person_age`                 | Age                              |
| `person_income`              | Annual Income                    |
| `person_home_ownership`      | Home ownership                   |
| `person_emp_length`          | Employment length (in years)     |
| `loan_intent`                | Loan intent                      |
| `loan_grade`                 | Loan grade                       |
| `loan_amnt`                  | Loan amount                      |
| `loan_int_rate`              | Interest rate                    |
| `loan_status`                | Loan status (0 is non-default, 1 is default) |
| `loan_percent_income`        | Percent income                   |
| `cb_person_default_on_file`  | Historical default               |
| `cb_person_cred_hist_length` | Credit history length            |

## 6. Exploratory Data Analysis (EDA)

### Data Overview

The dataset contains 32,581 entries with 12 columns. Below is a summary of the dataset:

- **RangeIndex:** 32581 entries, 0 to 32580
- **Data Columns:**
  - `person_age`: 32581 non-null, int64
  - `person_income`: 32581 non-null, int64
  - `person_home_ownership`: 32581 non-null, object
  - `person_emp_length`: 31686 non-null, float64
  - `loan_intent`: 32581 non-null, object
  - `loan_grade`: 32581 non-null, object
  - `loan_amnt`: 32581 non-null, int64
  - `loan_int_rate`: 29465 non-null, float64
  - `loan_status`: 32581 non-null, int64
  - `loan_percent_income`: 32581 non-null, float64
  - `cb_person_default_on_file`: 32581 non-null, object
  - `cb_person_cred_hist_length`: 32581 non-null, int64

### Summary Statistics

| Column                       | Count       | Mean        | Std         | Min         | 25%         | 50%         | 75%         | Max         |
|------------------------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| person_age                   | 32581       | 27.7346     | 6.3481      | 20.0        | 23.0        | 26.0        | 30.0        | 144.0       |
| person_income                | 32581       | 66074.85    | 61983.12    | 4000.0      | 38500.0     | 55000.0     | 79200.0     | 6000000.0   |
| person_emp_length            | 31686       | 4.7897      | 4.1426      | 0.0         | 2.0         | 4.0         | 7.0         | 123.0       |
| loan_amnt                    | 32581       | 9589.37     | 6322.09     | 500.0       | 5000.0      | 8000.0      | 12200.0     | 35000.0     |
| loan_int_rate                | 29465       | 11.0117     | 3.2405      | 5.42        | 7.90        | 10.99       | 13.47       | 23.22       |
| loan_status                  | 32581       | 0.2182      | 0.4130      | 0.0         | 0.0         | 0.0         | 0.0         | 1.0         |
| loan_percent_income          | 32581       | 0.1702      | 0.1068      | 0.0         | 0.09        | 0.15        | 0.23        | 0.83        |
| cb_person_cred_hist_length   | 32581       | 5.8042      | 4.0550      | 2.0         | 3.0         | 4.0         | 8.0         | 30.0        |

### Understanding Skewness in Numerical Features

The numerical features exhibit left skewness, where the majority of data points cluster towards lower values, with a tail extending towards higher values. This skewness can impact the model's ability to accurately describe typical cases. Typically occurring cases are more frequent in skewed datasets, whereas extreme cases are rare. Therefore, the model may prioritize accommodating rare occurrences at the expense of precision for more common scenarios.

For instance, determining a coefficient based on a thousand observations all within the range [0, 10] is likely more precise than for 990 observations in the same range and 10 observations ranging between [1,000, 1,000,000]. This imbalance can diminish the overall utility of your model.

### Benefits of Addressing Skewness

Addressing skewness can yield several advantages. It makes data analysis that relies on approximately Normal distributions more feasible and informative. It also ensures that results are reported on a sensible scale, although the relevance of this depends on the specific context. Additionally, correcting skewness helps prevent skewed predictors from disproportionately influencing predicted classifications compared to other predictors.

### Choosing a Transformation Approach

When addressing left skewness in numerical features, a log transformation is often preferred over standard scaling methods like MinMaxScaler or StandardScaler. The log transformation effectively reduces the impact of extreme values and compresses the range, making the data more symmetrical and normally distributed. This approach is particularly useful for variables such as incomes, populations, and other data that typically exhibit a log-normal distribution.

### Correlation Analysis

#### Correlation Heat Map

Here is the correlation matrix showing the relationships between different features:

|                          | person_age | person_income | person_emp_length | loan_amnt | loan_int_rate | loan_status | loan_percent_income | cb_person_cred_hist_length |
|--------------------------|------------|---------------|-------------------|-----------|---------------|-------------|---------------------|----------------------------|
| person_age               | 1.000000   | 0.173202      | 0.163106          | 0.050787  | 0.012580      | -0.021629   | -0.042411           | 0.859133                   |
| person_income            | 0.173202   | 1.000000      | 0.134268          | 0.266820  | 0.000792      | -0.144449   | -0.254471           | 0.117987                   |
| person_emp_length        | 0.163106   | 0.134268      | 1.000000          | 0.113082  | -0.056405     | -0.082489   | -0.054111           | 0.144699                   |
| loan_amnt                | 0.050787   | 0.266820      | 0.113082          | 1.000000  | 0.146813      | 0.105376    | 0.572612            | 0.041967                   |
| loan_int_rate            | 0.012580   | 0.000792      | -0.056405         | 0.146813  | 1.000000      | 0.335133    | 0.120314            | 0.016696                   |
| loan_status              | -0.021629  | -0.144449     | -0.082489         | 0.105376  | 0.335133      | 1.000000    | 0.379366            | -0.015529                  |
| loan_percent_income      | -0.042411  | -0.254471     | -0.054111         | 0.572612  | 0.120314      | 0.379366    | 1.000000            | -0.031690                  |
| cb_person_cred_hist_length | 0.859133  | 0.117987      | 0.144699          | 0.041967  | 0.016696      | -0.015529   | -0.031690           | 1.000000                   |

### Key Insights from Data Analysis

1. **Loan Interest Rate Missing Values:** The `loan_int_rate` feature contains missing values. These are addressed by imputing the missing values using a strategy based on the mean correlated with the target variable `loan_status`, where the correlation between `loan_int_rate` and `loan_status` is 0.34.

2. **Credit History Length and Age:** There is a high correlation (0.86) between `cb_person_cred_hist_length` and `person_age`, indicating that older individuals tend to have longer credit histories.

3. **Income and Loan Percent Income:** The `person_income` is negatively correlated (-0.25) with `loan_percent_income`, meaning higher income individuals tend to have a lower percentage of their income as loan amount.

4. **Employment Length and Loan Amount:** The `person_emp_length` has a weak positive correlation (0.11) with `loan_amnt`, suggesting that individuals with longer employment lengths tend to take higher loan amounts.

5. **Home Ownership Distribution:** The majority of loan applicants either rent (16,446) or have a mortgage (13,444), with fewer owning their homes outright (2,584) or having other forms of home ownership (107).

### Distribution of Categorical Features

#### Home Ownership

| Home Ownership | Count  |
|----------------|--------|
| RENT           | 16446  |
| MORTGAGE       | 13444  |
| OWN            | 2584   |
| OTHER          | 107    |

#### Loan Intent

| Loan Intent          | Count  |
|----------------------|--------|
| EDUCATION            | 6453   |
| MEDICAL              | 6071   |
| VENTURE              | 5719   |
| PERSONAL             | 5521   |
| DEBTCONSOLIDATION    | 5212   |
| HOMEIMPROVEMENT      | 3605   |

#### Loan Grade

| Loan Grade | Count  |
|------------|--------|
| A          | 10777  |
| B          | 10451  |
| C          | 6458   |
| D          | 3626   |
| E          | 964    |
| F          | 241    |
| G          | 64     |

#### Historical Default

| Historical Default | Count  |
|--------------------|--------|
| N                  | 26836  |
| Y                  | 5745   |

## Data Preprocessing

In the notebook `data_preparation.ipynb`, we prepare the data based on insights gained from the exploratory analysis (for details, see the notebook `exploratory_analysis`).

### Missing Values

Number of missing values per column:
- `person_age`: 0
- `person_income`: 0
- `person_home_ownership`: 0
- `person_emp_length`: 895
- `loan_intent`: 0
- `loan_grade`: 0
- `loan_amnt`: 0
- `loan_int_rate`: 3116
- `loan_status`: 0
- `loan_percent_income`: 0
- `cb_person_default_on_file`: 0
- `cb_person_cred_hist_length`: 0

### Steps in Data Preprocessing

1. **Imputation of Missing Values:**
   - Impute missing `loan_int_rate` values based on `loan_status`.
  ```
    mean_loan_int_rate_0 = data[data['loan_status'] == 0]['loan_int_rate'].mean()
    mean_loan_int_rate_1 = data[data['loan_status'] == 1]['loan_int_rate'].mean()
    data.loc[(data['loan_status'] == 0) & (data['loan_int_rate'].isnull()), 'loan_int_rate'] = mean_loan_int_rate_0
    data.loc[(data['loan_status'] == 1) & (data['loan_int_rate'].isnull()), 'loan_int_rate'] = mean_loan_int_rate_1
  ```
     
   - Impute missing `person_emp_length` values based on the overall mean.
     ```
      mean_person_emp_length = data['person_emp_length'].mean()
      data['person_emp_length'].fillna(mean_person_emp_length, inplace=True)
     ```

2. **One-Hot Encoding (OHE):**
   One-Hot-Encoding is a technique used to represent categorical variables as numerical values in a machine learning model. The following steps are taken:
   - Apply OHE to categorical columns with more than 2 categories (`loan_intent`, `loan_grade`, `person_home_ownership`).
     ```
       data = pd.get_dummies(data,columns=['loan_intent','loan_grade','person_home_ownership'])
     ```
   - Apply OHE to the categorical column with only 2 categories (`cb_person_default_on_file`), dropping the first category to avoid multicollinearity.
     ```
       data = pd.get_dummies(data,columns=['cb_person_default_on_file'], drop_first= True)
     ```

3. **Data Standardization:**
   We preprocess numerical columns by removing outliers and applying log transformation based on insights from the exploratory data analysis.
   - **Remove Outliers:** Identify and remove outliers to ensure a more normal distribution of data.
     ```
       column_no_outliers = data[column_name][data[column_name] < data[column_name].quantile(quantile)]
     ```
   - **Apply Log Transformation:** Apply log transformation to reduce skewness and make the data more normally distributed.
     ```
       column_log_transformed = np.log1p(column_no_outliers)
     ```

### Resulting Dataset

The resulting dataset contains 32,581 entries and 26 columns, with the following structure:

| Column                          | Non-Null Count | Dtype   |
|---------------------------------|----------------|---------|
| `person_age`                    | 32581          | float64 |
| `person_income`                 | 32581          | float64 |
| `person_emp_length`             | 32581          | float64 |
| `loan_amnt`                     | 32581          | float64 |
| `loan_int_rate`                 | 32581          | float64 |
| `loan_status`                   | 32581          | int64   |
| `loan_percent_income`           | 32581          | float64 |
| `cb_person_cred_hist_length`    | 32581          | float64 |
| `loan_intent_DEBTCONSOLIDATION` | 32581          | bool    |
| `loan_intent_EDUCATION`         | 32581          | bool    |
| `loan_intent_HOMEIMPROVEMENT`   | 32581          | bool    |
| `loan_intent_MEDICAL`           | 32581          | bool    |
| `loan_intent_PERSONAL`          | 32581          | bool    |
| `loan_intent_VENTURE`           | 32581          | bool    |
| `loan_grade_A`                  | 32581          | bool    |
| `loan_grade_B`                  | 32581          | bool    |
| `loan_grade_C`                  | 32581          | bool    |
| `loan_grade_D`                  | 32581          | bool    |
| `loan_grade_E`                  | 32581          | bool    |
| `loan_grade_F`                  | 32581          | bool    |
| `loan_grade_G`                  | 32581          | bool    |
| `person_home_ownership_MORTGAGE`| 32581          | bool    |
| `person_home_ownership_OTHER`   | 32581          | bool    |
| `person_home_ownership_OWN`     | 32581          | bool    |
| `person_home_ownership_RENT`    | 32581          | bool    |
| `cb_person_default_on_file_Y`   | 32581          | bool    |

The preprocessed dataset is saved as `processed_credit_risk_dataset.csv`.

## Model Training

In the notebook `model_training.ipynb`, we train various machine learning models to predict the likelihood of a loan applicant defaulting. The dataset used for training is the preprocessed dataset saved during the data preprocessing step.

### Reading the Dataset

We start by reading the preprocessed dataset.
````
data = pd.read_csv("../data/processed/processed_credit_risk_dataset.csv")
````

### Splitting the Data

The data is split into training and testing sets. Stratified sampling is used to handle the imbalance in the target labels, ensuring that the proportion of default and non-default cases is consistent across both sets.

```
y = data['loan_status'].to_numpy()
X = data.drop('loan_status',axis = 1).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state= 42)
```

### Metrics and Scoring

We define a function to evaluate our models using precision, recall, and F1-score metrics. These metrics provide a comprehensive view of the model's performance:
```
from sklearn.metrics import precision_recall_fscore_support
y_pred = model.predict(X_test)
    metrics = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
    
    print('Confusion matrix)\n', confusion_matrix(y_test, y_pred))
    print('Precision is {:0.2f} %'.format(metrics[0]*100))
    print('Recall is {:0.2f} %'.format(metrics[1]*100))
    print('Fscore is {:0.2f} %'.format(metrics[2]*100))
```

- **Precision:** The proportion of true positive results among the total predicted positives.
- **Recall:** The proportion of true positive results among the total actual positives.
- **F1-Score:** The harmonic mean of precision and recall, balancing the two metrics.

### Summary of Models Used

We trained several models, including:

- **Random Forest**
- **Decision Tree**
- **Logistic Regression** (with and without increased iterations)
- **K-Nearest Neighbors (KNN)**

#### Performance Metrics

**Precision:**

- **Random Forest:** 97.70%
- **Decision Tree:** 95.77%
- **Logistic Regression with increased iterations:** 76.17%
- **Logistic Regression:** 65.41%
- **KNN:** 54.82%

The Random Forest model has the highest precision, closely followed by the Decision Tree model.

**Recall:**

- **Random Forest:** 73.94%
- **Decision Tree:** 73.83%
- **Logistic Regression with increased iterations:** 54.14%
- **Logistic Regression:** 28.42%
- **KNN:** 27.52%

The Random Forest model also has the highest recall, indicating it has the lowest false negative rate.

**F1-Score:**

- **Random Forest:** 84.18%
- **Decision Tree:** 83.38%
- **Logistic Regression with increased iterations:** 63.29%
- **Logistic Regression:** 39.62%
- **KNN:** 36.64%

The Random Forest model achieves the highest F1-Score, which balances precision and recall effectively.

### Conclusion

Based on the precision, recall, and F1-Score metrics, the Random Forest model performs the best in predicting loan defaults. This model strikes a good balance between identifying positive cases (defaults) and minimizing false positives.

For more details and the complete code, refer to the notebook `model_training.ipynb`.

### Team Videos

1. [**Nestor Rojas**](https://youtu.be/iZWeZNim6cE)
2. [**Andriana Olashyn**](https://youtu.be/GpnisgeQ_50?si=-YaykzW35M8FLgl9)
3. [**Oleksandr Bratushchyk-Khoma**](https://youtu.be/W3-mOhDjcnE)
4. [**Veronika Plosnak**]()




