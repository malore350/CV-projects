import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import time

#output_file = open("rf.txt", 'w')

# Load the .mat file
data_mat = sio.loadmat('face.mat')
X = data_mat['X']

# Determine the number of identities (classes)
num_faces = 10  # number of images per identity
num_identities = X.shape[1] // num_faces

# Partition the data into training and testing sets
train_data = np.hstack([X[:, i*num_faces:i*num_faces+8] for i in range(num_identities)])
test_data = np.hstack([X[:, i*num_faces+8:(i+1)*num_faces] for i in range(num_identities)])

# Create labels for the dataset
labels = np.array([[i]*num_faces for i in range(num_identities)]).flatten()
train_labels = np.hstack([labels[i*num_faces:i*num_faces+8] for i in range(num_identities)])
test_labels = np.hstack([labels[i*num_faces+8:(i+1)*num_faces] for i in range(num_identities)])

# Function to train and evaluate the random forest classifier
def train_evaluate_rf(train_data, train_labels, test_data, test_labels, **params):
    # Initialize the Random Forest classifier with the given parameters
    rf = RandomForestClassifier(**params)

    # Start time for training
    start_time = time.time()
    # Train the classifier
    rf.fit(train_data.T, train_labels)
    # End time for training
    train_time = time.time() - start_time

    # Start time for testing
    start_time = time.time()
    # Predict the labels for the testing set
    test_predictions = rf.predict(test_data.T)
    # End time for testing
    test_time = time.time() - start_time

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, test_predictions)
    # Generate confusion matrix
    #conf_matrix = confusion_matrix(test_labels, test_predictions)

    # Store examples for visualization
    correct_idx = None
    incorrect_idx = None
    for i, (pred, actual) in enumerate(zip(test_predictions, test_labels)):
        if correct_idx is None and pred == actual:
            correct_idx = i
        if incorrect_idx is None and pred != actual:
            incorrect_idx = i
        if correct_idx is not None and incorrect_idx is not None:
            break

    return {
        'params': params,
        'accuracy': accuracy,
        #'conf_matrix': conf_matrix,
        'train_time': train_time,
        'test_time': test_time,
        'correct_idx': correct_idx,
        'incorrect_idx': incorrect_idx
    }

# Define the parameters to be tuned
parameters = {
    'n_estimators': [70, 100, 150, 200, 250],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'random_state': 42
}

# Store the results
results = []

# Track the best parameters and accuracy
best_params = None
best_accuracy = 0

# Iterate over all possible combinations of parameters using a nested loop
for n in parameters['n_estimators']:
    for depth in parameters['max_depth']:
        for split in parameters['min_samples_split']:
            for leaf in parameters['min_samples_leaf']:
                # Combine the current set of parameters
                current_params = {
                    'n_estimators': n,
                    'max_depth': depth,
                    'min_samples_split': split,
                    'min_samples_leaf': leaf,
                    'random_state': 42
                }
                
                # Train and evaluate the random forest classifier
                result = train_evaluate_rf(train_data, train_labels, test_data, test_labels, **current_params)
                print(f"Parameters: {result['params']}")
                print(f"Accuracy: {result['accuracy']:.4f}")
                print(f"Training time: {result['train_time']:.4f}s")
                print(f"Testing time: {result['test_time']:.4f}s")
                #print("Confusion Matrix:")
                #print(result['conf_matrix'])
                print("\n")
                # Append the result to the results list
                results.append(result)

                # Update the best parameters if the current accuracy is higher
                if result['accuracy'] > best_accuracy:
                    best_params = current_params
                    best_accuracy = result['accuracy']

# Print the best parameters
print(f"The best parameters based on accuracy are: {best_params} with an accuracy of {best_accuracy:.4f}")
