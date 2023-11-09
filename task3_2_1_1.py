import scipy.io as sio
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
from scipy.stats import mode
import seaborn as sns
from matplotlib import pyplot as plt

def plotting_close():
    plt.pause(3)
    plt.close()

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

def ensemble_predict(models, test_data):
    votes = []

    for pca_subspace, lda, pca_mean, nn_classifier in models:
        # Project the test sample onto the PCA subspace and transform with LDA
        X_test_pca = np.dot(test_data.T - pca_mean, pca_subspace)
        X_test_lda = lda.transform(X_test_pca)

        # Use KNN classifier to get the predicted class
        predictions = nn_classifier.predict(X_test_lda)
        votes.append(predictions)

    # Majority voting
    final_predictions = mode(np.array(votes), axis=0, keepdims = True)[0].flatten()
    return final_predictions

# Initial PCA to get all principal components
pca = PCA(n_components=min(train_data.shape[1], train_data.shape[0]-1), random_state=42)
pca.fit(train_data.T)

# Feature Randomization
# Parameters
T = 10  # Number of models in the ensemble
M_0 = 40  # Fixed number of largest eigenfaces
M_1 = 15  # Number of random eigenfaces
M_lda = 15 
bagging_factor = 1.0

#########################################################################################################
# WITH BAGGING
ensemble_models = []

for t in range(T):
    # Bootstrap sampling for bagging
    bootstrap_indices = resample(range(train_data.shape[1]), n_samples=int(train_data.shape[1]*bagging_factor))
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

    # Train KNN classifier on LDA-transformed data
    nn_classifier = KNeighborsClassifier(n_neighbors=1)
    nn_classifier.fit(lda.transform(X_bootstrap_pca), y_bootstrap)

    # Store the pca_subspace, lda object, pca mean, and KNN classifier
    ensemble_models.append((pca_subspace, lda, pca.mean_, nn_classifier))


# WITHOUT BAGGING
ensemble_models_no_bagging = []

for t in range(T):
    # Use entire training data without bootstrap sampling
    X_no_bagging = train_data
    y_no_bagging = train_labels

    # Select the fixed M_0 components
    pca_subspace_fixed = pca.components_[:M_0].T
    
    # Randomly select M_1 components from the remaining ones
    pca_subspace_random = pca.components_[M_0:][np.random.choice(range(pca.components_.shape[0] - M_0), M_1, replace=False)].T
    
    # Combine them to form the PCA subspace for this model
    pca_subspace = np.hstack((pca_subspace_fixed, pca_subspace_random))

    # Project the bootstrap sample onto the PCA subspace
    X_no_bagging_pca = np.dot(X_no_bagging.T - pca.mean_, pca_subspace)

    # Perform LDA on the PCA-reduced data
    lda = LDA(n_components=M_lda)
    lda.fit(X_no_bagging_pca, y_no_bagging)

    # Train KNN classifier on LDA-transformed data
    nn_classifier = KNeighborsClassifier(n_neighbors=1)
    nn_classifier.fit(lda.transform(X_no_bagging_pca), y_no_bagging)

    # Store the pca_subspace, lda object, pca mean, and KNN classifier
    ensemble_models_no_bagging.append((pca_subspace, lda, pca.mean_, nn_classifier))

# PREDICTING
# With bagging
# Predict with the ensemble
ensemble_predictions = ensemble_predict(ensemble_models, test_data)
ensemble_accuracy = accuracy_score(test_labels, ensemble_predictions)
ensemble_conf_matrix = confusion_matrix(test_labels, ensemble_predictions)
print(f'Ensemble Accuracy with bagging: {ensemble_accuracy}')

# Creating a confusion matrix with seaborn's heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(ensemble_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Ensemble Confusion Matrix')
plt.show()
plotting_close()

# Without bagging
# Predict with the ensemble
ensemble_predictions_no_bagging = ensemble_predict(ensemble_models_no_bagging, test_data)
ensemble_accuracy_no_bagging = accuracy_score(test_labels, ensemble_predictions_no_bagging)
ensemble_conf_matrix_no_bagging = confusion_matrix(test_labels, ensemble_predictions_no_bagging)
print(f'Ensemble Accuracy without bagging: {ensemble_accuracy_no_bagging}')

# Creating a confusion matrix with seaborn's heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(ensemble_conf_matrix_no_bagging, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Ensemble Confusion Matrix')
plt.show()

####################################################################################################
# WITHOUT RANDOIZATION IN SPACE
# Parameters
M_0 = 55  # Fixed number of largest eigenfaces
M_lda = 15  # Number of LDA components


# WITH BAGGING
ensemble_models_no_random = []

for t in range(T):
    # Bootstrap sampling for bagging
    bootstrap_indices = resample(range(train_data.shape[1]), n_samples=int(train_data.shape[1]*bagging_factor))
    X_bootstrap = train_data[:, bootstrap_indices]
    y_bootstrap = train_labels[bootstrap_indices]

    # Select the fixed M_0 components
    pca_subspace = pca.components_[:M_0].T

    # Project the bootstrap sample onto the PCA subspace
    X_bootstrap_pca = np.dot(X_bootstrap.T - pca.mean_, pca_subspace)

    # Perform LDA on the PCA-reduced data
    lda = LDA(n_components=M_lda)
    lda.fit(X_bootstrap_pca, y_bootstrap)

    # Train KNN classifier on LDA-transformed data
    nn_classifier = KNeighborsClassifier(n_neighbors=1)
    nn_classifier.fit(lda.transform(X_bootstrap_pca), y_bootstrap)

    # Store the pca_subspace, lda object, pca mean, and KNN classifier
    ensemble_models_no_random.append((pca_subspace, lda, pca.mean_, nn_classifier))


# WITHOUT BAGGING
ensemble_models_no_bagging_no_random = []

for t in range(T):
    # Use entire training data without bootstrap sampling
    X_no_bagging = train_data
    y_no_bagging = train_labels

    # Select the fixed M_0 components
    pca_subspace = pca.components_[:M_0].T

    # Project the bootstrap sample onto the PCA subspace
    X_no_bagging_pca = np.dot(X_no_bagging.T - pca.mean_, pca_subspace)

    # Perform LDA on the PCA-reduced data
    lda = LDA(n_components=M_lda)
    lda.fit(X_no_bagging_pca, y_no_bagging)

    # Train KNN classifier on LDA-transformed data
    nn_classifier = KNeighborsClassifier(n_neighbors=1)
    nn_classifier.fit(lda.transform(X_no_bagging_pca), y_no_bagging)

    # Store the pca_subspace, lda object, pca mean, and KNN classifier
    ensemble_models_no_bagging_no_random.append((pca_subspace, lda, pca.mean_, nn_classifier))


# PREDICTING
# With bagging
# Predict with the ensemble
ensemble_predictions = ensemble_predict(ensemble_models_no_random, test_data)
ensemble_accuracy = accuracy_score(test_labels, ensemble_predictions)
ensemble_conf_matrix = confusion_matrix(test_labels, ensemble_predictions)
print(f'Ensemble Accuracy with bagging, no feature space randomization: {ensemble_accuracy}')

# Creating a confusion matrix with seaborn's heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(ensemble_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Ensemble Confusion Matrix')
plt.show()


# Without bagging
# Predict with the ensemble
ensemble_predictions_no_bagging = ensemble_predict(ensemble_models_no_bagging_no_random, test_data)
ensemble_accuracy_no_bagging = accuracy_score(test_labels, ensemble_predictions_no_bagging)
ensemble_conf_matrix_no_bagging = confusion_matrix(test_labels, ensemble_predictions_no_bagging)
print(f'Ensemble Accuracy without bagging, no feature space randomization: {ensemble_accuracy_no_bagging}')

# Creating a confusion matrix with seaborn's heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(ensemble_conf_matrix_no_bagging, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Ensemble Confusion Matrix')
plt.show()


####################################################################################################

individual_accuracies = []
def calculate_ensemble_errors(ensemble_models, test_data, test_labels):
    individual_accuracies = []
    for pca_subspace, lda, pca_mean, _ in ensemble_models:
        # Project the test sample onto the PCA subspace
        X_test_pca = np.dot(test_data.T - pca_mean, pca_subspace)
        # Use LDA to get the accuracy
        individual_accuracy = lda.score(X_test_pca, test_labels)
        individual_accuracies.append(individual_accuracy)

    # Calculate the average sum-of-squares error for each individual model
    individual_sse = [(1 - accuracy)**2 for accuracy in individual_accuracies]

    # Calculate the average error of individual models (E_avg)
    E_avg = sum(individual_sse) / len(individual_sse)

    # Use the ensemble predict function to get ensemble accuracy
    ensemble_predictions = ensemble_predict(ensemble_models, test_data)
    ensemble_accuracy = accuracy_score(test_labels, ensemble_predictions)

    # Calculate the ensemble error as the square of (1 - ensemble accuracy)
    ensemble_sse = (1 - ensemble_accuracy)**2

    # Return the average error of individual models and the ensemble error
    return E_avg, ensemble_sse

# Example usage:
E_avg_bagging, E_com_bagging = calculate_ensemble_errors(ensemble_models, test_data, test_labels)
E_avg_no_bagging, E_com_no_bagging = calculate_ensemble_errors(ensemble_models_no_bagging, test_data, test_labels)
E_avg_bagging_no_random, E_com_bagging_no_random = calculate_ensemble_errors(ensemble_models_no_random, test_data, test_labels)
E_avg_no_bagging_no_random, E_com_no_bagging_no_random = calculate_ensemble_errors(ensemble_models_no_bagging_no_random, test_data, test_labels)
# Print results
print(f"Bagging - Average Error of Individual Models: {E_avg_bagging}, Committee Machine Error: {E_com_bagging}")
print(f"No Bagging - Average Error of Individual Models: {E_avg_no_bagging}, Committee Machine Error: {E_com_no_bagging}")
print(f"Bagging, No Randomization - Average Error of Individual Models: {E_avg_bagging_no_random}, Committee Machine Error: {E_com_bagging_no_random}")
print(f"No Bagging, No Randomization - Average Error of Individual Models: {E_avg_no_bagging_no_random}, Committee Machine Error: {E_com_no_bagging_no_random}")

