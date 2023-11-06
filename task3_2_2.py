import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Load data
data_path = 'face.mat'
data_mat = sio.loadmat(data_path)
X = data_mat['X']
num_faces = 10
num_identities = X.shape[1] // num_faces

# Partition the data into training and testing sets
train_data = np.hstack([X[:, i * num_faces:i * num_faces + 8] for i in range(num_identities)])
test_data = np.hstack([X[:, i * num_faces + 8:(i + 1) * num_faces] for i in range(num_identities)])
train_labels = np.array([[i] * 8 for i in range(num_identities)]).flatten()
test_labels = np.array([[i] * 2 for i in range(num_identities)]).flatten()

# Define the ensemble size and randomness parameter
ensemble_size = 20
randomness_parameter = 5  # Adjust this to your preferred level of randomness

# Define hyperparameter space
max_pca_components = min(100, train_data.shape[1])  # Assuming max 100 or number of features
max_lda_components = num_identities - 1  # Maximum number of LDA components is c-1 where c is the number of classes
pca_range = range(2, max_pca_components + 1)
lda_range = range(1, max_lda_components + 1)

best_M_pca = 55
best_M_lda = 15
# Ensemble training
ensemble_models = []
for i in range(ensemble_size):
    if i < randomness_parameter:  # Random models
        # Choose random PCA and LDA components
        pca_components = np.random.choice(pca_range)
        lda_components = min(np.random.choice(lda_range), pca_components - 1)
    else:  # Fixed models
        # Use the best known PCA and LDA components
        pca_components = best_M_pca
        lda_components = best_M_lda

    # Train PCA
    pca = PCA(n_components=pca_components)
    pca_train_data = pca.fit_transform(train_data.T)

    # Train LDA
    lda = LDA(n_components=lda_components)
    lda_train_data = lda.fit_transform(pca_train_data, train_labels)

    # Train NN classifier
    nn_classifier = KNeighborsClassifier(n_neighbors=1)
    nn_classifier.fit(lda_train_data, train_labels)

    # Store the trained model and its components
    ensemble_models.append((pca, lda, nn_classifier))



# Ensemble prediction
votes = []
for pca, lda, nn_classifier in ensemble_models:
    # Transform test data
    pca_test_data = pca.transform(test_data.T)
    lda_test_data = lda.transform(pca_test_data)

    # Predict with the current model
    predictions = nn_classifier.predict(lda_test_data)
    votes.append(predictions)

# Majority voting
votes = np.array(votes)
final_predictions, _ = mode(votes, axis=0, keepdims=True)
final_predictions = final_predictions.flatten()

# Evaluate ensemble accuracy
ensemble_accuracy = accuracy_score(test_labels, final_predictions)
print(f'Ensemble Accuracy: {ensemble_accuracy}')
