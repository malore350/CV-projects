# Re-import necessary libraries
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

# Function to perform PCA and LDA
def evaluate_pca_lda_nn(num_pca_components, num_lda_components, train_data, test_data, train_labels, test_labels):
    # Perform PCA with num_pca_components
    pca = PCA(n_components=num_pca_components, random_state=42)
    pca_train_data = pca.fit_transform(train_data.T)
    pca_test_data = pca.transform(test_data.T)
    
    # Perform LDA
    lda = LDA(n_components=num_lda_components)
    lda_train_data = lda.fit_transform(pca_train_data, train_labels)
    lda_test_data = lda.transform(pca_test_data)
    
    # Train NN classifier (1-Nearest Neighbor)
    nn_classifier = KNeighborsClassifier(n_neighbors=1)
    nn_classifier.fit(lda_train_data, train_labels)
    
    # Predict and evaluate
    predictions = nn_classifier.predict(lda_test_data)
    accuracy = accuracy_score(test_labels, predictions)
    
    return accuracy

# Grid search for optimal M_pca and M_lda
results = []
for M_pca in range(50, min(200, train_data.shape[1]), 1):  # Assuming train_data has samples as columns
    for M_lda in range(1, min(num_identities, M_pca+1)):
        accuracy = evaluate_pca_lda_nn(M_pca, M_lda, train_data, test_data, train_labels, test_labels)
        results.append((M_pca, M_lda, accuracy))


# Find the combination with the highest accuracy
best_M_pca, best_M_lda, best_accuracy = max(results, key=lambda x: x[2])

# Prepare data for 3D plot
M_pca_values, M_lda_values, accuracies = zip(*results)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for accuracy with respect to M_pca and M_lda
sc = ax.scatter(M_pca_values, M_lda_values, accuracies, c=accuracies, cmap='viridis')

# Labeling
ax.set_xlabel('M_pca')
ax.set_ylabel('M_lda')
ax.set_zlabel('Accuracy')
plt.title('3D Scatter Plot of PCA-LDA Accuracy')

# Color bar to show accuracy levels
colorbar = plt.colorbar(sc)
colorbar.set_label('Accuracy')

# Show the best combination
fig.text(0.05, 0.95, f"Best (M_pca, M_lda, Acc): ({best_M_pca}, {best_M_lda}, {best_accuracy:.2f})", 
         ha='left', va='top', color='red', weight='semibold')

plt.show()

# Return the best combination
print(best_M_pca, best_M_lda, best_accuracy)
