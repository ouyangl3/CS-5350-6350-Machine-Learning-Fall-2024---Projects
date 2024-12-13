import numpy as np
import pandas as pd
import csv

def numerical_to_binary(filepath, attribute_indices, first_column=True):
    if first_column:
        data = pd.read_csv(filepath)
    else:
        data = pd.read_csv(filepath).iloc[:, 1:]

    numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # Process numerical features
    for feature in numerical_features:
        column_name = data.columns[attribute_indices[feature]]
        median_value = data[column_name].median()
        data[column_name] = np.where(data[column_name].astype(float) > median_value, 1, 0)

    # Process categorical features
    for feature in attribute_indices:
        column_name = data.columns[attribute_indices[feature]]
        if data[column_name].dtype == 'object':
            majority_value = data[column_name][data[column_name] != '?'].mode()[0]
            data[column_name] = data[column_name].replace('?', majority_value)
            data[column_name] = pd.Categorical(data[column_name]).codes  # Encode as integers

    return data.values.tolist()

def load_and_preprocess_data(train_file, test_file, attribute_indices):
    train_data = numerical_to_binary(train_file, attribute_indices)
    test_data = numerical_to_binary(test_file, attribute_indices, False)
    return np.array(train_data), np.array(test_data)

def standardize_data(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, weights, l2_lambda=0.0):
    predictions = sigmoid(np.dot(X, weights))
    loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    if l2_lambda > 0:
        loss += l2_lambda * np.sum(weights**2) / 2
    return loss

def batch_gradient_descent(X, y, lr=0.01, epochs=1000, l2_lambda=0.0):
    weights = np.zeros(X.shape[1])
    losses = []

    for epoch in range(epochs):
        predictions = sigmoid(np.dot(X, weights))
        errors = predictions - y
        gradient = np.dot(X.T, errors) / len(y)
        if l2_lambda > 0:
            gradient += l2_lambda * weights
        weights -= lr * gradient

        # Track loss
        loss = compute_loss(X, y, weights, l2_lambda)
        losses.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss}")

    return weights, losses

def calculate_error_rate(predictions, examples, output_file="predictions.csv"):
    label_column = -1
    incorrect_predictions = 0

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Prediction"])
        
        for i, example in enumerate(examples):
            actual_label = example[label_column]
            predicted_probability = predictions[i]
            predicted_label = 1 if predicted_probability >= 0.5 else 0
            
            writer.writerow([i + 1, predicted_probability])
            
            if predicted_label != actual_label:
                incorrect_predictions += 1

    total_samples = len(examples)
    error_rate = incorrect_predictions / total_samples

    return error_rate


# Attribute indices
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

# Load and preprocess data
train_file = 'train_final.csv'
test_file = 'test_final.csv'

train_data, test_data = load_and_preprocess_data(train_file, test_file, attribute_indices)

# Separate features and labels
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test = test_data

# Standardize features
X_train = standardize_data(X_train)
X_test = standardize_data(X_test)

# Add bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Train model using batch gradient descent with L2 regularization
learning_rate = 0.01
epochs = 1000
l2_lambda = 0.01  # Regularization strength
weights, losses = batch_gradient_descent(X_train, y_train, lr=learning_rate, epochs=epochs, l2_lambda=l2_lambda)

# Evaluate model
train_predictions = sigmoid(np.dot(X_train, weights))
train_error = calculate_error_rate(train_predictions, train_data, output_file="train_predictions.csv")
print(f"Train Error: {train_error}")

test_predictions = sigmoid(np.dot(X_test, weights))
test_error = calculate_error_rate(test_predictions, test_data, output_file="test_predictions.csv")
print(f"Test Error: {test_error}")
