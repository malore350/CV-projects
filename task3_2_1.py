from sklearn.utils import resample
import scipy.io as sio
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from scipy.stats import mode
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns

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

# Initial PCA to get all principal components
pca = PCA(n_components=min(train_data.shape[1], train_data.shape[0]-1), random_state=42)
pca.fit(train_data.T)

# Parameters
T = 10  # Number of models in the ensemble
M_0 = 40  # Fixed number of largest eigenfaces
M_1 = 15  # Number of random eigenfaces
M_lda = 15  # Number of LDA components

ensemble_models = []

for t in range(T):
    # Bootstrap sampling for bagging
    bootstrap_indices = resample(range(train_data.shape[1]), n_samples=int(train_data.shape[1]*0.85))
    X_bootstrap = train_data[:, bootstrap_indices]
    y_bootstrap = train_labels[bootstrap_indices]

    # Select the fixed M_0 components
    pca_subspace_fixed = pca.components_[:M_0].T
    
    # Randomly select M_1 components from the remaining ones
    pca_subspace_random = pca.components_[M_0:][np.random.choice(range(pca.components_.shape[0] - M_0), M_1, replace=False)].T
    
    # Combine them to form the PCA subspace for this model
    pca_subspace = np.hstack((pca_subspace_fixed, pca_subspace_random))

    # Project the bootstrap sample onto the PCA subspace
    X_bootstrap_pca = np.dot(X_bootstrap.T - pca.mean_, pca_subspace)

    # Perform LDA on the PCA-reduced data
    lda = LDA(n_components=M_lda)
    lda.fit(X_bootstrap_pca, y_bootstrap)

    # Store the pca_subspace, lda object, and pca mean for later use
    ensemble_models.append((pca_subspace, lda, pca.mean_))


def ensemble_predict(models, test_data):
    votes = np.zeros((test_data.shape[1], len(models)), dtype=int)

    for i, (pca_subspace, lda, pca_mean) in enumerate(models):
        # Project the test sample onto the PCA subspace
        X_test_pca = np.dot(test_data.T - pca_mean, pca_subspace)

        # Use LDA to get the predicted class
        votes[:, i] = lda.predict(X_test_pca)
    
    # Majority voting
    final_predictions, _ = mode(votes, axis=1, keepdims=True)
    return final_predictions.flatten()

# Predict with the ensemble
ensemble_predictions = ensemble_predict(ensemble_models, test_data)

ensemble_accuracy = accuracy_score(test_labels, ensemble_predictions)
ensemble_conf_matrix = confusion_matrix(test_labels, ensemble_predictions)

print(f'Ensemble Accuracy: {ensemble_accuracy}')

# Creating a confusion matrix with seaborn's heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(ensemble_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Ensemble Confusion Matrix')
plt.show()




individual_accuracies = []
for pca_subspace, lda, pca_mean in ensemble_models:
    # Project the test sample onto the PCA subspace
    X_test_pca = np.dot(test_data.T - pca_mean, pca_subspace)
    # Use LDA to get the accuracy
    individual_accuracy = lda.score(X_test_pca, test_labels)
    individual_accuracies.append(individual_accuracy)

# Calculate the average sum-of-squares error for each individual model
individual_sse = [(1 - accuracy)**2 for accuracy in individual_accuracies]

# Calculate the average error of individual models (E_avg)
E_avg = sum(individual_sse) / len(individual_sse)

# Calculate the ensemble error as the square of (1 - ensemble accuracy)
ensemble_sse = (1 - ensemble_accuracy)**2

# Calculate the expected error of the committee machine (E_com)
# Assuming the errors are uncorrelated, which may not be true in practice
E_com = ensemble_sse

# Now you can print or compare E_avg and E_com
print(f"Average Error of Individual Models (E_avg): {E_avg}")
print(f"Expected Error of Committee Machine (E_com): {E_com}")

# If you want to compare E_com with the average of E_avg
# Here T is the number of models in the ensemble
E_com_average = E_avg / T
print(f"Expected Error of Committee Machine (averaged over models) (E_com_average): {E_com_average}")

