# DNA Methylation State Prediction

A Streamlit-based web application for predicting DNA methylation states using multiple machine learning models. The application allows users to train models on their own data or use pre-trained models to make predictions on new datasets.

## Features

- **Multiple ML Models**: Includes Logistic Regression, Random Forest, Gradient Boosting, and K-Nearest Neighbors classifiers
- **Comprehensive Analysis**: Provides detailed performance metrics and visualizations
- **Flexible Input**: Accepts CSV files for both training and testing
- **Model Persistence**: Save and load trained models for future use
- **Interactive UI**: User-friendly interface built with Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd EL
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run appp.py
   ```

2. Use the sidebar to:
   - Upload training and test datasets (CSV format)
   - Upload pre-trained models (optional)
   - Adjust model hyperparameters

3. View the predictions and model performance metrics in the main window

## File Structure

- `appp.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `*.pkl`: Pre-trained model files
- `train.csv`: Sample training data
- `test.csv`: Sample test data

## Data Format

The application expects CSV files with the following format:
- Each row represents a DNA methylation site
- The first column should contain the site identifiers
- The last column should contain the target variable (0/1 for unmethylated/methylated)
- Intermediate columns should contain the feature values

## Pre-trained Models

The repository includes several pre-trained models:
- `logistic_regression_model.pkl`: Logistic Regression model
- `random_forest_model.pkl`: Random Forest model
- `kneighbours_model.pkl`: K-Nearest Neighbors model
- `decision_tree_model.pkl`: Decision Tree model

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
