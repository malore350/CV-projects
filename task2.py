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

# Divide training data into 4 equal subsets
subset_size = train_data.shape[1] // 4
subsets = [train_data[:, i*subset_size:(i+1)*subset_size] for i in range(4)] # list of 4 subsets of shape (2576, 104)
subset_labels = [train_labels[i*subset_size:(i+1)*subset_size] for i in range(4)]

n_components = 104
# Incremental PCA
start_time = time.time()
ipca = IncrementalPCA(n_components=n_components)  # you can change n_components based on your needs

for subset in subsets:
    ipca.partial_fit(subset.T)

incremental_time = time.time() - start_time

# Batch PCA
start_time = time.time()
pca = PCA(n_components=n_components)  # same number of components as before
pca.fit(train_data.T)

batch_time = time.time() - start_time

# 1st Subset PCA
start_time = time.time()
first_subset_pca = PCA(n_components=n_components)
first_subset_pca.fit(subsets[0].T)

first_subset_time = time.time() - start_time

# Recognition
# Transform data
X_ipca = ipca.transform(test_data.T)
X_pca = pca.transform(test_data.T)
X_first_subset = first_subset_pca.transform(test_data.T)

# Create a k-NN classifier
knn = KNeighborsClassifier(n_neighbors=1)

# Incremental PCA
knn.fit(ipca.transform(train_data.T), train_labels)
accuracy_ipca = accuracy_score(test_labels, knn.predict(X_ipca))

# Batch PCA
knn.fit(pca.transform(train_data.T), train_labels)
accuracy_pca = accuracy_score(test_labels, knn.predict(X_pca))

# First Subset PCA
knn.fit(first_subset_pca.transform(subsets[0].T), subset_labels[0])
accuracy_first_subset = accuracy_score(test_labels, knn.predict(X_first_subset))

# Reconstruction
# Incremental PCA
X_ipca_reconstructed = ipca.inverse_transform(ipca.transform(test_data.T))
reconstruction_error_ipca = np.mean((test_data - X_ipca_reconstructed.T)**2)

# Batch PCA
X_pca_reconstructed = pca.inverse_transform(pca.transform(test_data.T))
reconstruction_error_pca = np.mean((test_data - X_pca_reconstructed.T)**2)

# First Subset PCA
X_first_subset_reconstructed = first_subset_pca.inverse_transform(first_subset_pca.transform(test_data.T))
reconstruction_error_first_subset = np.mean((test_data - X_first_subset_reconstructed.T)**2)


# Print results
print("Training time for Incremental PCA:", incremental_time)
print("Training time for Batch PCA:", batch_time)
print("Training time for First Subset PCA:", first_subset_time)

print("Accuracy for Incremental PCA:", accuracy_ipca)
print("Accuracy for Batch PCA:", accuracy_pca)
print("Accuracy for First Subset PCA:", accuracy_first_subset)

print("Reconstruction error for Incremental PCA:", reconstruction_error_ipca)
print("Reconstruction error for Batch PCA:", reconstruction_error_pca)
print("Reconstruction error for First Subset PCA:", reconstruction_error_first_subset)

# When we divide the training data into 4 equal subsets, the accuracy for Incremental PCA and Batch PCA are identical, which means that main data variances are well captured from the first batch.
# Dividing into more subsets will result in lower accuracy for Incremental PCA, because the first batch of data does not capture the main data variances well. Having 104 components is enough to capture the main data variances.