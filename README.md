
# Heart Disease Prediction Application

A web application that predicts the likelihood of heart disease based on patient health metrics. The application uses a machine learning model built with scikit-learn to analyze input parameters and provide a prediction with confidence level.

## Features

- **Prediction Interface**: Input patient health parameters and receive an instant prediction
- **Model Training**: Train the model with the latest dataset
- **Training Progress**: Visual feedback during the model training process
- **Detailed Metrics**: View model performance metrics and visualizations
  - Confusion Matrix
  - ROC Curve with AUC score
  - Learning Curve
  - Feature Correlation Matrix
  - Feature Distribution plots
  - Cross-Validation Accuracy

## Technology Stack

- **Backend**: Flask (Python)
- **ML Library**: scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **Data Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

## Getting Started

### Prerequisites

- Python 3.8+ (Compatible with Python 3.13)
- pip (Python package manager)

### Installation

1. Clone the repository
2. Navigate to the project directory
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

Run the application with:

```
python app_sklearn.py
```

Access the application at: http://127.0.0.1:5000/

## Usage

1. **Home Page**: Access the input form to enter patient parameters
2. **Training**: Click "Train Model" to train the model on the dataset
3. **Prediction**: Fill in the patient details and click "Predict" to get a prediction
4. **Metrics**: View detailed model metrics by clicking on "View Metrics"

## Model Information

The application uses the MLPClassifier (Multi-layer Perceptron) from scikit-learn, which is a neural network model. It is trained on the UCI Heart Disease dataset with the following parameters:

- Features include age, sex, chest pain type, resting blood pressure, cholesterol, and more
- The model is evaluated using cross-validation (3 folds)
- Performance metrics include accuracy, ROC AUC, confusion matrix, and feature importance

## Troubleshooting

- If you encounter Bootstrap loading issues, the application includes a fallback CSS file
- For prediction warnings, ensure all required fields are provided with appropriate values
- If training progress doesn't display, try refreshing the page after training completes

## License

This project is licensed under the MIT License - see the LICENSE file for details. 


