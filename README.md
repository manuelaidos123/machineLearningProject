# Machine Learning Model Comparison

## Overview

This project focuses on comparing the performance of different machine learning models on two datasets: "Credit" and "Adult." 
The goal is to evaluate the impact of various preprocessing techniques on model accuracy and provide insights into the strengths and weaknesses of each approach.

## Datasets

### Credit Dataset

The Credit dataset comprises 10,000 records and includes features related to individuals' credit histories. 
The dataset aims to predict the likelihood of credit approval for applicants based on various factors such as income, credit score, and debt-to-income ratio. 
The target variable is binary, indicating whether an applicant's credit application was approved (1) or denied (0).

### Adult Dataset

The Adult dataset is a collection of demographic and employment-related features for approximately 32,000 individuals. 
It is commonly used to predict whether an individual earns more than $50,000 per year, making it a binary classification task. The features include age, education level, occupation, and marital status. 
Challenges in this dataset include imbalanced class distribution, where a significant portion of individuals earns less than $50,000, making accurate predictions more complex. 
Additionally, the dataset includes both categorical and numerical features, requiring careful preprocessing strategies.

## Models

The following machine learning models were employed in the comparison:

- Naive Bayes
- KNeighbors
- Decision Tree
- Random Forest

## Preprocessing Techniques

The datasets were preprocessed using different techniques to assess their impact on model performance:

- **labelencoder:** Baseline encoding for categorical variables.
- **labelEncoder + standardScaler:** Combining label encoding and standard scaling.
- **labelEncoder + one-hot:** Combining label encoding and one-hot encoding.
- **labelencoder + one-hot + standard:** Combining label encoding, one-hot encoding, and standard scaling.

## Running the Code

1. Clone the repository:

   git clone [https://github.com/your-username/your-repo.git](https://github.com/manuelaidos123/machineLearningProject/tree/master)

   cd your-repo

3. Run the Project   

   pip install -r requirements.txt

   python main.py
   
