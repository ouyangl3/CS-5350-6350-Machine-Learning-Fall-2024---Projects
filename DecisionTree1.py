import csv
import math
import pandas as pd
import numpy as np
import csv
from collections import Counter, defaultdict

def numerical_to_binary(filepath, attribute_indices, first_column = True):
    if first_column:
        data = pd.read_csv(filepath)
    else:
        data = pd.read_csv(filepath).iloc[:, 1:]
    numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    for feature in numerical_features:
        column_name = data.columns[attribute_indices[feature]]
        median_value = data[column_name].median()

        # 1 if above median, 0 otherwise
        data[column_name] = np.where(data[column_name].astype(float) > median_value, 1, 0)

    return data.values.tolist()

def majority_label(examples, label_column):
    labels = [example[label_column] for example in examples]

    # Return the most common label
    return Counter(labels).most_common(1)[0][0]

def calculate_entropy(examples, label_column):
    each_label_count = Counter(example[label_column] for example in examples)
    total = len(examples)
    # Avoid division by zero
    if total == 0: 
        return 0

    entropy = -sum((count / total) * math.log2(count / total) for count in each_label_count.values())
    return entropy

def calculate_majority_error(examples, label_column):
    each_label_count = Counter(example[label_column] for example in examples)
    total = len(examples)
    if total == 0: 
        return 0

    majority = each_label_count.most_common(1)[0][1]
    majority_error = 1 - (majority / total)
    return majority_error

def calculate_gini(examples, label_column):
    each_label_count = Counter(example[label_column] for example in examples)
    total = len(examples)
    if total == 0: 
        return 0

    gini = 1 - sum((count/total)**2 for count in each_label_count.values())
    return gini

def calculate_measure(examples, method, label_column):
    if method == 'entropy':
        return calculate_entropy(examples, label_column)
    elif method == 'majority_error':
        return calculate_majority_error(examples, label_column)
    elif method == 'gini':
        return calculate_gini(examples, label_column)

    raise 'The method is invalid'

def best_attribute(examples, attributes, method, label_column):
    first_measure = calculate_measure(examples, method, label_column)

    best_gain = -1
    best_attribute = None

    for attribute in attributes:
        subsets = defaultdict(list)
        for example in examples:
            subsets[example[attribute]].append(example)

        total = len(examples)
        expected_measure = sum((len(subset) / total) * calculate_measure(subset, method, label_column) for subset in subsets.values())

        gain = first_measure - expected_measure
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute

    # Return the attribute that has the greatest gain
    return best_attribute

def get_all_possible_values_for_attribute(attribute_name):
    attribute_values = {
        'age': [0, 1],
        'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
        'fnlwgt': [0, 1],
        'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?'],
        'education-num': [0, 1],
        'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?'],
        'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'],
        'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?'],
        'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?'],
        'sex': ['Female', 'Male', '?'],
        'capital-gain': [0, 1],
        'capital-loss': [0, 1],
        'hours-per-week': [0, 1],
        'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']
    }
    return attribute_values.get(attribute_name, [])

def id3(examples, attributes, attribute_names, method, max_depth, current_depth=0):
    label_column = -1
    # If all examples have same label, return a leaf node with the label
    if len(set(example[label_column] for example in examples)) == 1:
        return examples[0][label_column]

    # If Attributes is empty or if the maximum depth has been reached, return a leaf node with the majority label
    if not attributes or current_depth == max_depth:
        return majority_label(examples, label_column)

    best_attribute_index = best_attribute(examples, attributes, method, label_column)
    best_attribute_name = attribute_names[best_attribute_index]

    tree = {best_attribute_name: {}}

    all_possible_values = get_all_possible_values_for_attribute(best_attribute_name)

    for value in all_possible_values:
        subset = [example for example in examples if example[best_attribute_index] == value]

        # For each value, recurse to create subtrees
        if subset:
            # Remove the best attribute for further splits
            attributes = attributes - {best_attribute_index}
            tree[best_attribute_name][value] = id3(subset, attributes, attribute_names, method, max_depth, current_depth + 1)
        else:
            # If no examples have this value, use the majority label
            tree[best_attribute_name][value] = majority_label(examples, label_column)

    return tree

def predict(tree, example, attribute_indices):
    # Base case: if the current node is a leaf node, return its value
    if not isinstance(tree, dict):
        return tree

    # Get the first key in the dictionary as the attribute
    attribute = next(iter(tree))
    attribute_value = example[attribute_indices[attribute]]
    
    # Check if the attribute value has a corresponding subtree
    if attribute_value in tree[attribute]:
        subtree = tree[attribute][attribute_value]
    
    return predict(subtree, example, attribute_indices)

def calculate_error_rate(data, tree, attribute_indices, output_file="predictions.csv"):
    label_column = -1
    incorrect_predictions = 0
    predictions = []

    for idx, example in enumerate(data, start=1):
        actual_label = example[label_column]
        predicted_label = predict(tree, example, attribute_indices)
        
        predictions.append([idx, predicted_label])

        if predicted_label != actual_label:
            incorrect_predictions += 1

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Prediction"])
        writer.writerows(predictions)
    total = len(data)
    error_rate = incorrect_predictions / total
    return error_rate

attribute_names = {
    0: 'age',
    1: 'workclass',
    2: 'fnlwgt',
    3: 'education',
    4: 'education-num',
    5: 'marital-status',
    6: 'occupation',
    7: 'relationship',
    8: 'race',
    9: 'sex',
    10: 'capital-gain',
    11: 'capital-loss',
    12: 'hours-per-week',
    13: 'native-country'
}

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

train_data = numerical_to_binary('train_final.csv', attribute_indices)
test_data = numerical_to_binary('test_final.csv', attribute_indices, False)
attributes = set(range(14))

max_depth = 14
decision_tree = id3(train_data, attributes, attribute_names, 'entropy', max_depth)
# print(decision_tree)
error = calculate_error_rate(test_data, decision_tree, attribute_indices)
print(f'{max_depth}. The Average Prediction Error Rate: {error:.4f}')

# methods = ['entropy', 'majority_error', 'gini']
# for method in methods:
#     print(method)
#     for max_depth in range(1,14):
#         decision_tree = id3(train_data, attributes, attribute_names, method, max_depth)
#         # print(decision_tree)
#         error = calculate_error_rate(test_data, decision_tree, attribute_indices)
#         print(f'{max_depth}. The Average Prediction Error Rate: {error:.4f}')
