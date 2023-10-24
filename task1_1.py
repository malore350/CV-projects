import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# TASK A
# Load the .mat file
data_mat = sio.loadmat('face.mat')
X = data_mat['X']

num_faces = 10  # number of images per identity
num_identities = X.shape[1] // num_faces
print(f'Number of identities: {num_identities}')

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

# Compute the covariance matrix
S = (A @ A.T) / num_identities
rank_S = np.linalg.matrix_rank(S)
print(rank_S)

# Calculate eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(S)
sorted_indices = np.argsort(eigenvalues)[::-1]      # sort eigenvalues in descending order
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

# Visualizing the mean face and the first few eigenfaces
plt.figure(figsize=(5, 5))
plt.imshow(mean_face_pca, cmap='gray')
plt.title('Mean Face')
plt.axis('off')
plt.show()


fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    eigenface = eigenvectors[:, i].reshape(56, 46)
    axes[i].imshow(eigenface, cmap='gray')
    axes[i].set_title(f'Eigenface {i+1}')
    axes[i].axis('off')
plt.tight_layout()
plt.show()


# Plot the sorted eigenvalues
plt.figure(figsize=(10, 6))
plt.plot(eigenvalues)
plt.xlabel('Eigenvector Index')
plt.ylabel('Eigenvalue Magnitude')
plt.title('Eigenvalues in Descending Order')
plt.grid(True)
plt.show()

# We can pick rank_S eigenfaces to represent the data. After that, the eigenvalues are ~ 0. So, we can pick 416 eigenfaces to represent the data.
# We have rank(S) eigenvectors with nonzero eigenvalues.


# TASK B
# Compute the alternate covariance matrix
S_2 = (A.T @ A) / num_identities

# Calculate eigenvectors and eigenvalues for the alternate covariance matrix
eigenvalues_2, eigenvectors_2 = np.linalg.eigh(S_2)
sorted_indices_2 = np.argsort(eigenvalues_2)[::-1]  # sort eigenvalues in descending order
eigenvalues_2 = eigenvalues_2[sorted_indices_2]
eigenvectors_2 = eigenvectors_2[:, sorted_indices_2]

# Transform the eigenvectors back to the original space
eigenvectors_transformed = A @ eigenvectors_2
# Normalize the transformed eigenvectors
eigenvectors_transformed = eigenvectors_transformed / np.linalg.norm(eigenvectors_transformed, axis=0)

# Visualizing the first few eigenfaces for the alternate method
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    eigenface = eigenvectors_transformed[:, i].reshape(56, 46)
    axes[i].imshow(eigenface, cmap='gray')
    axes[i].set_title(f'Eigenface {i+1} (Alternate)')
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# Plot the sorted eigenvalues for the alternate method
plt.figure(figsize=(10, 6))
plt.plot(eigenvalues_2)
plt.xlabel('Eigenvector Index (Alternate)')
plt.ylabel('Eigenvalue Magnitude (Alternate)')
plt.title('Eigenvalues in Descending Order (Alternate)')
plt.grid(True)
plt.show()

# Comparing the eigenvalues from both methods
plt.figure(figsize=(10, 6))
plt.plot(eigenvalues, label='From S')
plt.plot(eigenvalues_2, label='From S_alternate')
plt.xlabel('Eigenvector Index')
plt.ylabel('Eigenvalue Magnitude')
plt.title('Comparison of Eigenvalues')
plt.legend()
plt.grid(True)
plt.show()


# Initialize an empty list to hold the cosine similarities
cosine_similarities = []

# Calculate cosine similarity for the first five pairs of eigenvectors
for i in range(5):
    cos_sim = np.dot(eigenvectors[:, i], eigenvectors_transformed[:, i])
    cosine_similarities.append(cos_sim)

# Print the cosine similarities
print("Cosine Similarities for the first five pairs of eigenvectors:", cosine_similarities)

# Cosine Similarities for the first five pairs of eigenvectors: [-1.0000000000000004, -1.0000000000000009, 0.9999999999999988, 0.9999999999999991, 1.0]
# The cosine similarities are not exactly 1 because of numerical errors.
# Negative cosine similarity means they are exactly on the same vector, just pointing in the opposite direction.
# For this reason we can see that the images are very similar, just the pixels are inverted (where it is dark it becomes light and vice versa)