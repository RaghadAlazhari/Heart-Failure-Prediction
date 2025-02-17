# Heart Failure Prediction

## Overview

This project aims to predict the likelihood of heart failure based on various medical attributes using machine learning algorithms. The dataset includes details about patients' health, such as age, sex, chest pain type, cholesterol levels, and other features related to heart health. By analyzing this data, the project applies several machine learning techniques to predict the presence or absence of heart disease, contributing to early diagnosis and better health outcomes.

## Dataset

The dataset used in this project contains the following attributes:

- **Age**: Age of the patient.
- **Sex**: Gender of the patient.
- **ChestPainType**: Type of chest pain.
- **RestingBP**: Resting blood pressure in mm Hg.
- **Cholesterol**: Serum cholesterol in mg/dl.
- **FastingBS**: Fasting blood sugar level.
- **RestingECG**: Resting electrocardiographic results.
- **MaxHR**: Maximum heart rate achieved.
- **ExerciseAngina**: Whether the patient has exercise-induced angina.
- **Oldpeak**: Depression induced by exercise relative to rest.
- **ST_Slope**: Slope of the peak exercise ST segment.
- **HeartDisease**: Target variable.

## Installation

To get started with this project, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/RaghadAlazhari/heart-failure-prediction.git
```

### Dependencies

The following libraries are required for this project:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **matplotlib**: For visualizations.
- **seaborn**: For statistical data visualizations.
- **scikit-learn**: For machine learning models and evaluations.

Install the necessary libraries.

## Usage

### 1. **Data Analysis**

First, analyze the dataset to understand the structure, clean missing values, and visualize the data. Libraries used for analysis:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

- Load the dataset using `pandas`.
- Use `seaborn` and `matplotlib` to visualize data and find correlations.

### 2. **Model Training**

In this step, you will train the model using different machine learning classifiers, including:

- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- **Naive Bayes**
- **Support Vector Classifier (SVC)**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

- Split the dataset into training and testing sets using `train_test_split`.
- Train the models using the classifiers above.
- Evaluate the models with a confusion matrix and classification report.

### 3. **Evaluation**

Evaluate each model's performance based on key metrics like:

- **Confusion Matrix**
- **Classification Report** (precision, recall, f1-score)

## License
This project is open-source and available under the MIT License.
For more details, visit my GitHub profile: [RaghadAlazhari](https://github.com/RaghadAlazhari/)


