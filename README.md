# Employee Attrition Prediction Using ANN

## Project Overview
This project uses an Artificial Neural Network (ANN) to predict whether an employee is likely to leave a company based on employee-related information.

The goal is to explore a practical binary-classification problem, prepare tabular data, train a neural network, and evaluate its predictions.

## Dataset
- **Dataset:** IBM HR Analytics Employee Attrition
- **Rows:** 1,470 employees
- **Target column:** `Attrition`
- **Classes:** `Yes` (employee leaves) and `No` (employee stays)

The target was encoded as:
- `No` → `0`
- `Yes` → `1`

The dataset has an imbalanced target distribution, so the project uses stratified data splitting and class weights during training to help the model pay more attention to the minority class.

## Workflow
1. Load and inspect the dataset.
2. Explore the data, including attrition distribution and relationships between employee attributes and attrition.
3. Prepare the features and target.
4. Remove constant columns and the employee identifier.
5. Convert categorical features into numeric features using one-hot encoding with `pandas.get_dummies()`.
6. Split the data into training and testing sets using stratification.
7. Scale the features using `StandardScaler`.
8. Build and train an ANN using TensorFlow/Keras.
9. Use dropout, early stopping, and class weights to help manage overfitting and class imbalance.
10. Evaluate predictions using accuracy, a confusion matrix, precision, recall, and F1-score.
11. Try predictions on an individual employee example.

## Model
The model is a feed-forward neural network for binary classification. It uses dense layers with ReLU activation in the hidden layers and a sigmoid output layer to estimate the probability of attrition.

The probability is converted to a class prediction using a threshold of `0.5`:
- Probability below `0.5` → `No`
- Probability greater than or equal to `0.5` → `Yes`

## Evaluation Results
Results from the current experiment:

| Metric | Result |
|---|---:|
| Test accuracy | 79.93% |
| Test loss | 0.4433 |
| Validation accuracy (reported run) | 81.36% |

For the class-wise results from the current test evaluation:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `0` — No attrition | 0.91 | 0.84 | 0.88 | 247 |
| `1` — Attrition | 0.41 | 0.57 | 0.48 | 47 |

**Note:** Accuracy alone is not enough for an imbalanced dataset. The recall and F1-score for the attrition class are especially important because they show how well the model identifies employees who may leave. These results are from one experiment and are not a guarantee of real-world performance.

## Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Repository Files
- `Employee_Attrition_Prediction_ANN.ipynb` — project notebook
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` — dataset
- `employee_attrition_model.keras` — saved trained model
- `employee_attrition_scaler.pkl` — saved feature scaler

**Important:** For predictions on new employees, use the same feature columns, one-hot encoding layout, and scaling procedure used during training. The scaler alone does not store the feature-column order or categorical encoding mapping, so keep those preprocessing details with the project.

## How to Run
1. Clone or download this repository.
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib jupyter
   ```
3. Open the notebook in Jupyter:
   ```bash
   jupyter notebook
   ```
4. Run the notebook cells in order.

## Learning Outcomes
- Exploratory data analysis
- Handling categorical features and imbalanced classes
- Train/validation/test splitting
- Feature scaling without fitting on validation or test data
- Building and evaluating a binary-classification ANN
- Understanding overfitting, dropout, and early stopping
- Interpreting confusion matrices and classification reports

## Disclaimer
This is an educational machine-learning project. Predictions should not be used as the sole basis for employment decisions. Employee data is sensitive, and real-world use would require careful privacy, fairness, legal, and ethical review.
