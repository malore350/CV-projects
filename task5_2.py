# based on task5_1.py, based on this original task and best parameters ({'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 1}) can you please write a new py file that will plot a confusion matrix for this rf, and also visualize correct and wrong predictions (3 each) showing predicted and actual label
# make a plot of classification report

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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
    conf_matrix = confusion_matrix(test_labels, test_predictions)

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
    
    correct_indices = []
    incorrect_indices = []
    for i, (pred, actual) in enumerate(zip(test_predictions, test_labels)):
        if pred == actual:
            correct_indices.append(i)
        else:
            incorrect_indices.append(i)
        if len(correct_indices) >= 3 and len(incorrect_indices) >= 3:
            break

    return {'accuracy': accuracy, 'conf_matrix': conf_matrix, 'correct_idx': correct_indices, 'incorrect_idx': incorrect_indices, 'train_time': train_time, 'test_time': test_time, 'test_predictions': test_predictions}

# Define the parameters to be tested
params = {
    'n_estimators': 250,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

# Train and evaluate the classifier
results = train_evaluate_rf(train_data, train_labels, test_data, test_labels, **params)

# Print the results
print(f"Accuracy: {results['accuracy']}")
print(f"Training time: {results['train_time']} seconds")
print(f"Testing time: {results['test_time']} seconds")

# Plotting the confusion matrix using seaborn for a more informative visualization
plt.figure(figsize=(10, 8))
sns.heatmap(results['conf_matrix'], annot=True, fmt='g', cmap='viridis')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot examples of 3 correct predictions of different classes in one plot

# Ensure the correct and incorrect indices are lists and limit to 3 each
correct_indices = results['correct_idx'][:3] if results['correct_idx'] else []
incorrect_indices = results['incorrect_idx'] if results['incorrect_idx'] else []
incorrect_indices_18 = [idx for idx in incorrect_indices if test_labels[idx] == 18][:1]
incorrect_indices_29 = [idx for idx in incorrect_indices if test_labels[idx] == 29][:1]

# # # Visualizing correct predictions of 3 different classes (1, 2, 3), make this a separate plot
# plt.figure(figsize=(15, 5))
# class_indices = [np.where(test_labels == i)[0][0] for i in range(1, 4)]
# for i, idx in enumerate(class_indices):
#     plt.subplot(1, 3, i+1)
#     plt.imshow(test_data[:, idx].reshape(46, 56).T, cmap='gray')
#     plt.title(f"Correct: True: {test_labels[idx]}, Pred: {results['test_predictions'][idx]}")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

# # Visualizing incorrect predictions of 3 different classes and class 24 train example image next to it
# plt.figure(figsize=(15, 5))

# # Include class 24 along with incorrect predictions for 18 and 29
# class_indices = [np.where(test_labels == i)[0][0] for i in [18, 29, 24]]  # 24 added

# for i, idx in enumerate(class_indices):
#     plt.subplot(1, 3, i + 1)  # Adjust subplot for three images
#     plt.imshow(test_data[:, idx].reshape(46, 56).T, cmap='gray')
#     if i < 2:  # For classes 18 and 29, show incorrect predictions
#         plt.title(f"Incorrect: True: {test_labels[idx]}, Pred: {results['test_predictions'][idx]}")
#     else:  # For class 24, just show the class label
#         plt.title(f"Class: {test_labels[idx]}")
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

plt.figure(figsize=(15, 5))
class_indices = [np.where(test_labels == i)[0][0] for i in range(1, 4)]
labels = ['(a)', '(b)', '(c)']  # New labels

for i, idx in enumerate(class_indices):
    plt.subplot(1, 3, i+1)
    plt.imshow(test_data[:, idx].reshape(46, 56).T, cmap='gray')
    #plt.title(labels[i])  # Use the new label
    plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))
class_indices = [np.where(test_labels == i)[0][0] for i in [18, 29, 24]]  # 24 added
labels = ['(a)', '(b)', '(c)']  # New labels

for i, idx in enumerate(class_indices):
    plt.subplot(1, 3, i + 1)  # Adjust subplot for three images
    plt.imshow(test_data[:, idx].reshape(46, 56).T, cmap='gray')
    #plt.title(labels[i])  # Use the new label
    plt.axis('off')

plt.tight_layout()
plt.show()


# Plot the classification report
y_pred = results['test_predictions']
print(classification_report(test_labels, y_pred))
