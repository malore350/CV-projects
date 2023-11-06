# Re-import necessary libraries
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import numpy as np

# Reload the .mat file since execution state was reset
data_path = 'face.mat'  # Update path to the location of the uploaded file
data_mat = sio.loadmat(data_path)
X = data_mat['X']

num_faces = 10  # number of images per identity
num_identities = X.shape[1] // num_faces

# Partition the data into training and testing sets
train_data = np.hstack([X[:, i * num_faces:i * num_faces + 8] for i in range(num_identities)])
test_data = np.hstack([X[:, i * num_faces + 8:(i + 1) * num_faces] for i in range(num_identities)])

# Labels for training and test data
train_labels = np.array([[i] * 8 for i in range(num_identities)]).flatten()
test_labels = np.array([[i] * 2 for i in range(num_identities)]).flatten()

best_M_pca = 55

# Perform PCA with best_M_pca
pca = PCA(n_components=best_M_pca, random_state=42)
pca_train_data = pca.fit_transform(train_data.T)

# The means of each class (projected into the PCA subspace)
class_means = np.array([np.mean(pca_train_data[train_labels == i], axis=0) for i in range(num_identities)])

# The overall mean of the data (projected into the PCA subspace)
overall_mean = np.mean(pca_train_data, axis=0)

# Initialize within-class scatter matrix S_W
S_W = np.zeros((best_M_pca, best_M_pca))

# Calculate within-class scatter matrix S_W
for i in range(num_identities):
    class_scatter = np.cov(pca_train_data[train_labels == i], rowvar=False, bias=True)
    S_W += class_scatter

# Initialize between-class scatter matrix S_B
S_B = np.zeros((best_M_pca, best_M_pca))

# Calculate between-class scatter matrix S_B
for i in range(num_identities):
    mean_diff = (class_means[i] - overall_mean).reshape(best_M_pca, 1)
    S_B += np.outer(mean_diff, mean_diff.T)

# Calculate the rank of the scatter matrices
within_class_scatter_matrix_rank = np.linalg.matrix_rank(S_W)
between_class_scatter_matrix_rank = np.linalg.matrix_rank(S_B)

print(f'Rank of within-class scatter matrix: {within_class_scatter_matrix_rank}')
print(f'Rank of between-class scatter matrix: {between_class_scatter_matrix_rank}')
