# Predicting Accident Severity Using Machine Learning

## Overview
This project aims to predict the severity of road accidents using machine learning techniques, focusing on the impact of weather conditions, road infrastructure, and temporal factors. The dataset used for this analysis is the [US Accidents Dataset (2016 - 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents), specifically filtered for accidents occurring in the state of Florida.

By leveraging traditional machine learning models, this project seeks to provide insights that can help policymakers and traffic authorities take proactive measures to improve road safety.

## Dataset
The original dataset contains accident records from across the United States, including detailed information on weather conditions, road infrastructure, and accident severity. However, for this project, only data from **Florida** is used.

Users must **manually filter** Florida-specific accident records from the original dataset before running the analysis. The dataset includes:
- **Severity Levels**: The severity of accidents categorized into four levels.
- **Weather Conditions**: Temperature, humidity, visibility, precipitation, and general weather status.
- **Road Infrastructure**: Features such as traffic signals, crossings, and road bumps.
- **Temporal Features**: Day/Night classification and Twilight Phase indicators.

## Features Used
The key features considered in this project for accident severity prediction include:
- **Distance (mi)** – The length of the road segment where the accident occurred.
- **Temperature (F)** – Ambient temperature at the time of the accident.
- **Humidity (%)** – Air moisture level.
- **Visibility (mi)** – Distance a driver can see under normal conditions.
- **Precipitation (in)** – Rainfall at the time of the accident.
- **Weather Condition** – General weather description (e.g., clear, fog, rain).
- **Road Infrastructure Features** – Presence of traffic signals, crossings, junctions, etc.
- **Twilight Phase** – Whether the accident occurred during twilight conditions.

## Machine Learning Approach
### Data Preprocessing
- **Handling Missing Data**: Missing values for features such as temperature, precipitation, and visibility were imputed based on logical groupings.
- **Class Imbalance Handling**: The dataset exhibited a significant class imbalance, which was addressed using **SMOTE (Synthetic Minority Oversampling Technique)**.
- **Feature Engineering**: A new binary feature, **Twilight Phase**, was engineered to consolidate different twilight-related columns into a single feature.

### Model Training
- The dataset was split into **80% training and 20% testing**.
- The **Random Forest Classifier** was chosen for its robustness and interpretability.
- **Hyperparameter tuning** was performed using **GridSearchCV** and **StratifiedKFold**.

### Evaluation Metrics
The model's performance was assessed using:
- **F1-Score (Macro and Weighted Averages)**
- **Confusion Matrix**
- **Cross-Validation** (5-fold) to ensure generalization across different subsets of data.

## Results
The trained **Random Forest Classifier** achieved:
- **88.00% Accuracy**
- **84.57% Mean F1 Macro Score** across 5-fold cross-validation.

The most influential features in predicting accident severity were:
1. **Distance (mi)** – 37.63%
2. **Temperature (F)** – 17.36%
3. **Humidity (%)** – 15.22%
4. **Traffic Signal** – 4.17%
5. **Crossing** – 2.47%

## Acknowledgments
- The dataset used in this project is provided by **Sobhan Moosavi** on Kaggle.
- This study builds upon prior research in accident severity prediction and machine learning techniques.

