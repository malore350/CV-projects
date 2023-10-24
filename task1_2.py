import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import time

import seaborn as sns

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
num_bases_list = [5, 100, 200, 300, 414, 1000]  # Number of bases to use for reconstruction
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
num_bases = 416  # Number of bases to use for reconstruction
# Project training and test data onto the PCA bases
train_features = (train_data - mean_face_pca.flatten()[:, np.newaxis]).T @ eigenvectors[:, :num_bases]
test_features = (test_data - mean_face_pca.flatten()[:, np.newaxis]).T @ eigenvectors[:, :num_bases]

# Labels for training and test data
train_labels = np.array([[i] * 8 for i in range(num_identities)]).flatten()
test_labels = np.array([[i] * 2 for i in range(num_identities)]).flatten()

# Step 2: Classification
# Initialize and train k-NN classifier
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_features, train_labels)

# Predict labels for test data
predicted_labels = knn.predict(test_features)
end_time = time.time()

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
