import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Load the .mat file using mat73
data_mat73 = sio.loadmat('face.mat')
X = data_mat73['X']

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

#plot mean centered face
plt.figure(figsize=(5, 5))
plt.imshow(mean_centered_image, cmap='gray')
plt.title('Mean Centered Face')
plt.axis('off')
plt.show()


# Flatten the mean-centered data for PCA
A = mean_centered_data.reshape(mean_centered_data.shape[0], -1).T

# Compute the covariance matrix
S = (A @ A.T) / num_identities
rank_S = np.linalg.matrix_rank(S)
print(rank_S)

# Calculate eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(S)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Visualizing the mean face and the first few eigenfaces
# plt.figure(figsize=(5, 5))
# plt.imshow(mean_face_pca, cmap='gray')
# plt.title('Mean Face')
# plt.axis('off')
# plt.show()


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

# We can pick rank_S - 1 eigenfaces to represent the data. After that, the eigenvalues are ~ 0. So, we can pick 414 eigenfaces to represent the data.

