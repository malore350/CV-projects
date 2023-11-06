import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
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

# Best Values
best_M_pca = 55
best_M_lda = 15

# Perform PCA with best_M_pca
pca = PCA(n_components=best_M_pca, random_state=42)
pca_train_data = pca.fit_transform(train_data.T)
pca_test_data = pca.transform(test_data.T)

# Perform LDA
lda = LDA(n_components=best_M_lda)
lda_train_data = lda.fit_transform(pca_train_data, train_labels)
lda_test_data = lda.transform(pca_test_data)

# Train NN classifier (1-Nearest Neighbor)
nn_classifier = KNeighborsClassifier(n_neighbors=1, p = 2)
nn_classifier.fit(lda_train_data, train_labels)

# Predict and evaluate
predictions = nn_classifier.predict(lda_test_data)
accuracy = accuracy_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)

print(f'Accuracy: {accuracy}')

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()


# Find success case
success_index = np.where(test_labels == predictions)[0][0]  # Just take the first success

# Find failure case
failure_index = np.where(test_labels != predictions)[0][0]  # Just take the first failure

# Now we'll plot the success and failure cases
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Success case
success_image = test_data[:, success_index].reshape(46, 56).T
axes[0].imshow(success_image, cmap='gray')
axes[0].set_title(f"Success\nTrue: {test_labels[success_index]}\nPred: {predictions[success_index]}")
axes[0].axis('off')

# Failure case
failure_image = test_data[:, failure_index].reshape(46, 56).T
axes[1].imshow(failure_image, cmap='gray')
axes[1].set_title(f"Failure\nTrue: {test_labels[failure_index]}\nPred: {predictions[failure_index]}")
axes[1].axis('off')

plt.tight_layout()
plt.show()
