# Predicting Airline Customer Satisfaction

**Author:** Hannah Canon

## Repository Structure
``` bash
your_project/
├── data/                # Raw and processed data
├── images/              # Images used for README.md
├── README.md            # Project documentation
└── requirements.txt     # Required dependencies
```

## Project Overview

This project aims to predict **airline customer satisfaction** using a set of passenger and flight-related features such as flight distance, seat comfort, inflight wifi service, and more. The dataset contains various attributes about passengers and their flights, which are used to determine satisfaction levels.

## Dataset

- **Source**: Airline Passenger Satisfaction Data
- **Size**: ~129,880 rows, 24 features
- **Features**: Includes demographic information, flight details, and satisfaction level.
  - **Examples**:
    - `Flight Distance`
    - `Class`
    - `Inflight wifi service`
    - `Seat comfort`
    - `Arrival Delay in Minutes`
    - `Customer satisfaction` (target variable)

## Objective

The goal is to build a predictive model to classify whether a passenger is satisfied with their flight experience based on various factors such as service quality, delays, and onboard comfort.

## Methodology

1. **Data Cleaning**: Addressed missing values using KNNImputer, handled outliers, and corrected inconsistencies.
2. **Visualization**: Created heatmaps, pair plots, and correlation matrices to identify trends and relationships.
3. **Preprocessing**: Applied scaling, one-hot encoding (OHE), and balanced the dataset using SMOTE (Synthetic Minority Over-sampling Technique).
4. **Model Training**: Trained LightGBM and Decision Tree models, with hyperparameter tuning to optimize performance.
5. **Evaluation**: Assessed models using accuracy, precision, recall, F1 score, and confusion matrices to gauge prediction quality.
6. **Deployment**: Deployed the model using  app.py to serve predictions on new data through a simple user interface.

## Results

- 
  
## Conclusion

The model successfully identifies key factors influencing customer satisfaction, particularly in-flight comfort and services such as online boarding and flight class. These insights can be used by airlines to improve the passenger experience.

## Future Work

- Implement more advanced models such as Random Forest or Gradient Boosting.
- Analyze further customer segments (e.g., business vs. leisure travelers).

## Installation ⚙️

# Clone the repo
```bash
git clone https://github.com/hannahcanon09/ML-classification-app
```
# Nagivate to the project directory
```bash
cd ML-classification-app
```

## Contact 📧
Hannah Canon - hannahacanon@gmail.com

Project link: https://github.com/hannahcanon09/ML-classification-app
