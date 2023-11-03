import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from scipy import stats
import time

import seaborn as sns

import psutil

# Measure memory usage
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # rss: Resident Set Size, in bytes


# PC-LDA
# Load the .mat file
data_mat = sio.loadmat('face.mat')
X = data_mat['X']

num_faces = 10  # number of images per identity
num_identities = X.shape[1] // num_faces

# Partition the data into training and testing sets using mat73-loaded data
train_data = np.hstack([X[:, i*num_faces:i*num_faces+8] for i in range(num_identities)])
test_data = np.hstack([X[:, i*num_faces+8:(i+1)*num_faces] for i in range(num_identities)])

# Labels for training and test data
train_labels = np.array([[i] * 8 for i in range(num_identities)]).flatten()
test_labels = np.array([[i] * 2 for i in range(num_identities)]).flatten()

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




M_pca = 191  # Vary this
W_pca = eigenvectors[:, :M_pca]  # Select first M_pca eigenvectors
train_pca = W_pca.T @ train_data
test_pca = W_pca.T @ test_data

# Step 2: LDA
M_lda = 51  # Vary this
lda = LinearDiscriminantAnalysis(n_components=M_lda)
train_lda = lda.fit_transform(train_pca.T, train_labels)
test_lda = lda.transform(test_pca.T)

# Generate labels
train_labels = np.array([i for i in range(num_identities) for _ in range(8)])
test_labels = np.array([i for i in range(num_identities) for _ in range(2)])

# Step 3: k-NN Classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_lda, train_labels)

# Step 4: Evaluate performance
predictions = knn.predict(test_lda)
accuracy = accuracy_score(test_labels, predictions)

conf_matrix = confusion_matrix(test_labels, predictions)

# Report
print("Accuracy:", accuracy)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Scatter matrix Rank
# Calculate mean vectors for each class
mean_vectors = []
for cl in range(0, num_identities):
    mean_vectors.append(np.mean(train_lda[train_labels == cl], axis=0))

# Calculate within-class scatter matrix
S_W = np.zeros((M_lda, M_lda))
for cl, mv in zip(range(0, num_identities), mean_vectors):
    class_sc_mat = np.zeros((M_lda, M_lda))
    for row in train_lda[train_labels == cl]:
        row, mv = row.reshape(M_lda, 1), mv.reshape(M_lda, 1)
        class_sc_mat += (row - mv).dot((row - mv).T)
    S_W += class_sc_mat

# Calculate between-class scatter matrix
overall_mean = np.mean(train_lda, axis=0)
S_B = np.zeros((M_lda, M_lda))
for i, mean_vec in enumerate(mean_vectors):
    n = train_lda[train_labels == i, :].shape[0]
    mean_vec = mean_vec.reshape(M_lda, 1)
    overall_mean = overall_mean.reshape(M_lda, 1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

# Calculate ranks
rank_S_W = np.linalg.matrix_rank(S_W)
rank_S_B = np.linalg.matrix_rank(S_B)

print("Rank of within-class scatter matrix:", rank_S_W)
print("Rank of between-class scatter matrix:", rank_S_B)





# PC-LDA Ensemble

# Initialize lists to store errors
committee_errors = []
individual_errors = []

# Number of base models
n_base_models = 10

# Randomness parameter for feature space
random_feature_ratio = 0.8

# Ensemble through Bagging
bagging_knn = BaggingClassifier(estimator=knn, n_estimators=n_base_models, max_samples=0.8)

base_models = []
selected_features_list = []

for i in range(n_base_models):
    
    # Randomly select features (PCA components)
    selected_features = np.random.choice(range(M_pca), int(M_pca * random_feature_ratio), replace=False)
    selected_features_list.append(selected_features)
    
    # Train PCA-LDA with bagging
    bagging_model = clone(bagging_knn)
    bagging_model.fit(train_pca[selected_features, :].T, train_labels)
    base_models.append(bagging_model)

    
    # Compute individual errors
    individual_pred = bagging_model.predict(test_pca[selected_features, :].T)
    # individual_errors.append(1 - accuracy_score(test_labels, individual_pred))
    
    # Compute individual sum-of-squares errors (approximating E[{y_t(x) - h(x)}^2])
    individual_sse = np.sum((individual_pred - test_labels)**2) / len(test_labels)
    individual_errors.append(individual_sse)


# Fusion rules: Voting
preds = np.array([model.predict(test_pca[selected_features, :].T) for model, selected_features in zip(base_models, selected_features_list)])
final_pred = stats.mode(preds, keepdims=True)[0][0]

# Calculate the error of the committee machine
committee_error = 1 - accuracy_score(test_labels, final_pred)
committee_errors.append(committee_error)

# Calculate the average error of individual models
avg_individual_error = np.mean(individual_errors)

# Confusion Matrix
conf_matrix_ensemble = confusion_matrix(test_labels, final_pred)

# Calculate recognition accuracy
recognition_accuracy = accuracy_score(test_labels, final_pred)

print(f'Recognition Accuracy: {recognition_accuracy * 100:.2f}%')
print(f'Error of Committee Machine: {committee_error}')
print(f'Average Error of Individual Models: {avg_individual_error}')
sns.heatmap(conf_matrix_ensemble, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()