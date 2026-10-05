import numpy as np
import pandas as pd

def calculate_accuracy(true_labels, predicted_labels):
    true_labels = np.asarray(true_labels).flatten()
    predicted_labels = np.asarray(predicted_labels).flatten()
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predicted_labels)
    
    if not (set(unique_true) <= {1, -1}) or not (set(unique_pred) <= {1, -1}):
        raise ValueError("Labels must be +1 or -1")
    
    correct_predictions = np.sum(true_labels == predicted_labels)
    total_samples = len(true_labels)
    accuracy = (correct_predictions / total_samples) * 100
    
    return accuracy

def min_max_normalize(matrix):
    min_vals = np.min(matrix, axis=0) 
    max_vals = np.max(matrix, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    normalized_matrix = (matrix - min_vals) / range_vals
    return normalized_matrix
import numpy as np

def load_dataset(csv_file,unlabeled_ratio=0.15, test_ratio=0.4):
    
    data = np.genfromtxt(csv_file, delimiter=",", dtype=str, skip_header=1)
    class_names = np.unique(data[:, -1])
    print(f"classes: {class_names[0]} / {class_names[1]}")
    print(f"dataset samples: {data.shape[0]} / features: {data.shape[1] - 1}")
    if class_names[0] in np.unique(data[:, -1]) or class_names[1] in np.unique(data[:, -1]):
        data[:, -1] = np.where(data[:, -1] == class_names[0], 1, -1)

    data = data.astype(np.float32)

    features = min_max_normalize(data[:, :-1])
    

    np.random.seed(10000)
    indices = np.random.permutation(len(features))

    split_idx = int(len(features) * (1 - unlabeled_ratio))
    labeled_test_features = features[indices[:split_idx]]
    labeled_test_labels = data[indices[:split_idx]][:, -1]
    U = features[indices[split_idx:]]

    test_split_idx = int(len(labeled_test_features) * (1 - test_ratio))
    X = labeled_test_features[:test_split_idx]
    y = labeled_test_labels[:test_split_idx]
    X_test = labeled_test_features[test_split_idx:]
    y_test = labeled_test_labels[test_split_idx:]

    return X, y, X_test, y_test, U


def move_labels_to_last_column(file_dir, from_column):
    df = pd.read_csv(file_dir)  

    col_to_move = df.columns[from_column]  
    df_reordered = df[[col for col in df.columns if col != col_to_move] + [col_to_move]]

    df_reordered.to_csv(file_dir, index=False)