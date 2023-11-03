import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import time

import seaborn as sns

import psutil

# Measure memory usage
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # rss: Resident Set Size, in bytes


# TASK A
# Load the .mat file
data_mat = sio.loadmat('face.mat')
X = data_mat['X']

num_faces = 10  # number of images per identity
num_identities = X.shape[1] // num_faces

# Partition the data into training and testing sets using mat73-loaded data
train_data = np.hstack([X[:, i*num_faces:i*num_faces+8] for i in range(num_identities)])
test_data = np.hstack([X[:, i*num_faces+8:(i+1)*num_faces] for i in range(num_identities)])

# Reshape the training data images
train_images = np.array([img.reshape(46, 56).T for img in train_data.T])

# Compute the mean face for PCA
mean_face_pca = np.mean(train_images, axis=0)

# Mean-center the data
mean_centered_data = train_images - mean_face_pca
mean_centered_image = np.mean(mean_centered_data, axis=0)

# Flatten the mean-centered data for PCA
A = mean_centered_data.reshape(mean_centered_data.shape[0], -1).T

S_2 = (A.T @ A) / num_identities

# Calculate eigenvectors and eigenvalues for the alternate covariance matrix
eigenvalues_2, eigenvectors_2 = np.linalg.eigh(S_2)
sorted_indices_2 = np.argsort(eigenvalues_2)[::-1]  # sort eigenvalues in descending order
eigenvalues_2 = eigenvalues_2[sorted_indices_2]
eigenvectors_2 = eigenvectors_2[:, sorted_indices_2]

# Transform the eigenvectors back to the original space
eigenvectors = A @ eigenvectors_2
# Normalize the transformed eigenvectors
eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)


# Function to reconstruct image
def reconstruct_image(image, eigenvectors, mean_face, num_bases):
    # Step 1: Mean-center the image
    mean_centered_image = image - mean_face
    
    # Step 2: Project the mean-centered image onto the PCA bases
    projection = np.dot(mean_centered_image.flatten(), eigenvectors[:, :num_bases])
    
    # Step 3: Reconstruct the image
    reconstruction = np.dot(eigenvectors[:, :num_bases], projection) + mean_face.flatten()
    
    # Reshape to original dimensions
    reconstructed_image = reconstruction.reshape(mean_face.shape)
    
    # Step 4: Calculate reconstruction error
    error = np.sum((image - reconstructed_image)**2)
    
    return reconstructed_image, error

# Sample usage
num_bases_list = [5, 100, 200, 300, 415, 1000]  # Number of bases to use for reconstruction
sample_images = [train_images[0], test_data[:, 0].reshape(46, 56).T, train_images[-1]]  # Sample images from training and testing datasets
# sample_images = [test_data[:, 0].reshape(46, 56).T, test_data[:, 2].reshape(46, 56).T, test_data[:, 4].reshape(46, 56).T]

# Loop through each sample image and reconstruct it using different numbers of bases
for i, image in enumerate(sample_images):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(num_bases_list) + 1, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    for j, num_bases in enumerate(num_bases_list):
        reconstructed_image, error = reconstruct_image(image, eigenvectors, mean_face_pca, num_bases)
        plt.subplot(1, len(num_bases_list) + 1, j + 2)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title(f'Reconstructed\n{num_bases} bases\nError: {error:.2f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# TASK B
# TRAINING with NN
num_bases = 415  # Number of bases to use for reconstruction
# Project training and test data onto the PCA bases
train_features = (train_data - mean_face_pca.flatten()[:, np.newaxis]).T @ eigenvectors[:, :num_bases]
test_features = (test_data - mean_face_pca.flatten()[:, np.newaxis]).T @ eigenvectors[:, :num_bases]

# Labels for training and test data
train_labels = np.array([[i] * 8 for i in range(num_identities)]).flatten()
test_labels = np.array([[i] * 2 for i in range(num_identities)]).flatten()

# Step 2: Classification
# Initialize and train k-NN classifier
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', p=1)
knn.fit(train_features, train_labels)
# Using algorithms ball_tree and kd_tree for kNN is faster than brute force/auto
# Using p = 1 (Manhattan distance) is more accurate than p = 2 (Euclidean distance)
# Check for mem usage with leaves

# Predict labels for test data
predicted_labels = knn.predict(test_features)
end_time = time.time()

memory_usage = get_memory_usage()   # check the memory usage
print(f'Final memory usage: {memory_usage:.2f} MB')

# Step 3: Evaluation
# Calculate recognition accuracy
accuracy = accuracy_score(test_labels, predicted_labels)

# Generate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Time taken
time_taken = end_time - start_time

print(f'Accuracy: {accuracy:.2f}')
print(f'Time taken: {time_taken:.2f} seconds')

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Select specific indices from your test set (replace these with the indices of the faces you're interested in)
specific_indices = [0, 10, 20]  # For example, to test on the first, 11th, and 21st faces in your test set

# Select corresponding features and labels
specific_test_features = test_features[specific_indices]
specific_test_labels = test_labels[specific_indices]

# Make predictions using k-NN for these specific faces
specific_predictions = knn.predict(specific_test_features)

# If you want to show these specific test faces
for i, index in enumerate(specific_indices):
    specific_face_image = test_data[:, index].reshape(46, 56).T
    plt.subplot(1, len(specific_indices), i+1)
    plt.imshow(specific_face_image, cmap='gray')
    if specific_test_labels[i] == specific_predictions[i]:
        plt.title(f"Label: {specific_test_labels[i]}\nPrediction: {specific_predictions[i]}\nSuccessfull Classification")
    else:
        plt.title(f"Label: {specific_test_labels[i]}\nPrediction: {specific_predictions[i]}\nFailed Classification")
    plt.axis('off')

plt.tight_layout()
plt.show()
