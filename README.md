# Project: Credit Risk Prediction

## 1. Project Definition and Objective

**Objective:** Develop a machine learning model to predict the likelihood of a loan applicant defaulting.

**Goal:** Create a robust model to help financial institutions assess the risk associated with granting loans.

## 2. Main Question

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

## 3. Data Collection

**Source:** [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

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
| `loan_status`                | Loan status (0 is non default 1 is default) |
| `loan_percent_income`        | Percent income                   |
| `cb_person_default_on_file`  | Historical default               |
| `cb_person_cred_hist_length` | Credit history length            |

## 4. Exploratory Data Analysis (EDA)

- **Visualizations:** Analyze distributions of borrower characteristics, loan amounts, and outcomes.
- **Statistics:** Compute summary statistics for numeric and categorical features.
- **Correlation Analysis:** Check the correlation between different features and the target variable (default).
