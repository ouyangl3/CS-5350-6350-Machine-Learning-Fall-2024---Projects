import numpy as np
import pandas as pd
from scipy.optimize import minimize
import csv
from tqdm import tqdm

def save_predictions_to_csv(predictions, output_file="test_predictions.csv"):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Prediction"])
        for i, pred in enumerate(predictions):
            writer.writerow([i + 1, pred])
    print(f"Predictions saved to {output_file}")

def numerical_to_binary(filepath, attribute_indices, first_column=True):
    if first_column:
        data = pd.read_csv(filepath)
    else:
        data = pd.read_csv(filepath).iloc[:, 1:]

    numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    for feature in numerical_features:
        column_name = data.columns[attribute_indices[feature]]
        median_value = data[column_name].median()
        data[column_name] = np.where(data[column_name].astype(float) > median_value, 1, 0)

    for feature in attribute_indices:
        column_name = data.columns[attribute_indices[feature]]
        if data[column_name].dtype == 'object':
            majority_value = data[column_name][data[column_name] != '?'].mode()[0]
            data[column_name] = data[column_name].replace('?', majority_value)
            data[column_name] = pd.Categorical(data[column_name]).codes

    return data.values.tolist()

def load_and_preprocess_data(train_file, test_file, attribute_indices):
    train_data = numerical_to_binary(train_file, attribute_indices)
    test_data = numerical_to_binary(test_file, attribute_indices, False)
    return np.array(train_data), np.array(test_data)

def standardize_data(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std

attribute_indices = {
    'age': 0,
    'workclass': 1,
    'fnlwgt': 2,
    'education': 3,
    'education-num': 4,
    'marital-status': 5,
    'occupation': 6,
    'relationship': 7,
    'race': 8,
    'sex': 9,
    'capital-gain': 10,
    'capital-loss': 11,
    'hours-per-week': 12,
    'native-country': 13
}

train_file = 'train_final.csv'
test_file = 'test_final.csv'
train_data, test_data = load_and_preprocess_data(train_file, test_file, attribute_indices)

x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_train, y_train = x_train[:3000], y_train[:3000]
x_test = test_data

x_train = standardize_data(x_train)
x_test = standardize_data(x_test)

y_train = np.where(y_train == 0, -1, 1)

def kernel_function(x1, x2):
    return np.dot(x1, x2.T)

def svm_dual_svm(X, y, C):
    n_samples = X.shape[0]

    K = np.dot(X, X.T)
    
    def objective(alpha):
        return 0.5 * np.dot(alpha * y, np.dot(K, alpha * y)) - np.sum(alpha)

    constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
    bounds = [(0, C) for _ in range(n_samples)]
    initial_alpha = np.zeros(n_samples)

    result = minimize(
        objective,
        initial_alpha,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100}
    )

    alpha = result.x

    w = np.dot((alpha * y), X)

    support_vector_indices = np.where((alpha > 1e-5) & (alpha < C - 1e-5))[0]
    if len(support_vector_indices) == 0:
        support_vector_indices = np.where(alpha > 1e-5)[0]
    b = np.mean([y[i] - np.dot(w, X[i]) for i in support_vector_indices])

    return w, b, alpha

def decision_to_probability(decision_values):
    return 1 / (1 + np.exp(-decision_values))

C_values = [3]
for C in C_values:
    print(f"Training with C = {C:.4f}")
    w_dual, b_dual, alpha = svm_dual_svm(x_train, y_train, C)
    print(f"Weights (dual): {w_dual}")
    print(f"Bias (dual): {b_dual}")

    train_predictions_dual = np.sign(np.dot(x_train, w_dual) + b_dual)

    train_error_dual = np.mean(train_predictions_dual != y_train)
    print(f"Training Error (dual): {train_error_dual:.4f}")

    test_predictions_dual = decision_to_probability(np.dot(x_test, w_dual) + b_dual)

    save_predictions_to_csv(test_predictions_dual, f"test_predictions_C_{C:.4f}.csv")
