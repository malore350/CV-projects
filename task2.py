import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import IncrementalPCA, PCA
import time

# TASK A
# Load the .mat file
data_mat = sio.loadmat('face.mat')
X = data_mat['X']

num_faces = 10  # number of images per identity
num_identities = X.shape[1] // num_faces

# Partition the data into training and testing sets using mat73-loaded data
train_data = np.hstack([X[:, i*num_faces:i*num_faces+8] for i in range(num_identities)])
test_data = np.hstack([X[:, i*num_faces+8:(i+1)*num_faces] for i in range(num_identities)])

# Labels for the training and testing sets
train_labels = np.array([[i] * 8 for i in range(num_identities)]).flatten()
test_labels = np.array([[i] * 2 for i in range(num_identities)]).flatten()

# Number of principal components to keep
num_bases = 104

# Initialize variables for tracking metrics
batch_train_times = []
inc_train_times = []
batch_accuracies = []
inc_accuracies = []
batch_reconstruction_errors = []
inc_reconstruction_errors = []

# Function to perform classification and get reconstruction error
def evaluate_pca(train_features, test_features, train_labels, test_labels):
    # Initialize and train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_features, train_labels)
    # Predict labels for test data
    predicted_labels = knn.predict(test_features)
    # Calculate recognition accuracy
    accuracy = accuracy_score(test_labels, predicted_labels)
    # Compute reconstruction error on a sample test image
    sample_test_image = test_data[:, 0]
    reconstruction = np.dot(test_features[0, :], pca.components_) + pca.mean_
    reconstruction_error = np.sum((sample_test_image - reconstruction)**2)
    return accuracy, reconstruction_error

# Loop through subsets of the data
for end_idx in range(104, 417, 104):
    subset = train_data[:, :end_idx]

    # Batch PCA
    start_time = time.time()
    pca = PCA(n_components=num_bases)
    pca.fit(subset.T)
    batch_train_time = time.time() - start_time
    batch_train_times.append(batch_train_time)
    
    # Project data onto the principal components
    batch_train_features = pca.transform(subset.T)
    batch_test_features = pca.transform(test_data.T)
    
    # Evaluation for Batch PCA
    batch_accuracy, batch_reconstruction_error = evaluate_pca(batch_train_features, batch_test_features, train_labels[:end_idx], test_labels)
    batch_accuracies.append(batch_accuracy)
    batch_reconstruction_errors.append(batch_reconstruction_error)

    # Incremental PCA
    ipca = IncrementalPCA(n_components=num_bases, batch_size=104)
    start_time = time.time()
    for i in range(0, end_idx, 104):
        ipca.partial_fit(subset[:, i:i+104].T)
    inc_train_time = time.time() - start_time
    inc_train_times.append(inc_train_time)
    
    # Project data onto the principal components
    inc_train_features = ipca.transform(subset.T)
    inc_test_features = ipca.transform(test_data.T)
    
    # Evaluation for Incremental PCA
    inc_accuracy, inc_reconstruction_error = evaluate_pca(inc_train_features, inc_test_features, train_labels[:end_idx], test_labels)
    inc_accuracies.append(inc_accuracy)
    inc_reconstruction_errors.append(inc_reconstruction_error)

# Results
print("Batch PCA Training Times:", batch_train_times)
print("Incremental PCA Training Times:", inc_train_times)
print("Batch PCA Accuracies:", batch_accuracies)
print("Incremental PCA Accuracies:", inc_accuracies)
print("Batch PCA Reconstruction Errors:", batch_reconstruction_errors)
print("Incremental PCA Reconstruction Errors:", inc_reconstruction_errors)