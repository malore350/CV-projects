---
## Abstract:
  This report presents an implementation and evaluation of Eigenfaces
  using Principal Component Analysis (PCA), Incremental PCA, and Linear
  Discriminant Analysis (LDA) for face recognition. The effectiveness of
  ensemble learning through LDA and the optimization of Random Forest
  classifiers are also explored. Each method's performance is critically
  assessed by examining accuracy, error rates, and computational
  efficiency. The findings contribute to a deeper understanding of the
  practical applications of these algorithms in the field of computer
  vision.
## Author:
- Kamran Gasimov<sup>1</sup> and Alexandra Suchshenko<sup>2</sup> [^1] [^2]
## " **CS485 Machine Learning for Computer Vision: Coursework 1** "
---

# NAIVE EIGENFACE APPROACH

## Approach

The approach employed throughout this work, the Eigenface approach,
treats the images as linearized vectors of an image, which takes
$D\times1$ space, where $D = W\times H$. A method to distill the
essential information from a face image involves identifying the primary
axes of variation within a set of face images and using these axes to
encode and differentiate between individual faces. Face images are
mapped onto a feature space defined by the most significant variations
across the set of images. These pivotal features are termed eigenfaces,
which are essentially the principal components or eigenvectors of the
face dataset. These eigenfaces might not align with identifiable facial
features like eyes or mouth. To recognize a face, a new image is
projected onto the space defined by the eigenfaces, and its identity is
determined by comparing its coordinates in this \"face space\" with
those of known faces, which is done using Principal Component Analysis
(PCA). \[1\]

## Dataset

The experiment utilized a dataset comprising 520 facial images, each
with dimensions of 46 by 56 pixels (which makes the number of features
of every vector to be 2576), categorized into 52 distinct classes with
10 images per class. The dataset was divided into training and testing
sets with a ratio of 8:2, resulting in 416 images for training and 104
for testing. Data has been divided in a way that train/test datasets do
not share a common class, to remove bias in the data. Normalization of
the training images was achieved by deducting the average face image,
calculated as the mean across all images, from each image, later used
across several applications. This process involved no additional data
augmentation or transformation.

## Eigenfaces

The mean face in Fig. 1 has been calculated by finding the mean of the
training data ($\bar{x} = \sum\limits_{i=1}^{N}X_i$).

![Mean Face]({"mean_image"}.png){#fig:mean_face width="3.5cm"}

In PCA, eigenvectors (eigenfaces) of the covariance matrix represent the
directions of maximum variance in the data. Each eigenface is a vector
that, when reshaped to the dimensions of the original images, looks like
a face but with certain exaggerated or diminished features. The
eigenvalues correspond to the magnitude of the variance captured by each
eigenvector. Only a limited number of eigenvalues have non-zero numbers,
which are determined by the rank of the covariance matrix *S*. S can be
calculated by using either one of
$$S_1=\frac{1}{N} AA^T = S_2 = \frac{1}{N} A^TA$$ The covariance matrix
itself is a measure of how changes in one pixel are linearly associated
with changes in other pixels across the set of images. We decided to use
415 eigenvectors having the largest eigenvalues to represent the whole
data. However, we do not have to use all of them for the task.

![Eigenvalues in Descending
Order]({"eigen_plot_zoomed"}.png){#fig:reconstruct_1 width="8cm"}

Although it is difficult to assess the required number of eigenvectors
due to ambiguity in the threshold, Fig. 2 shows that the curve stagnates
at around 300.

The difference in the eigenvectors between the methods of calculation of
*S* was mainly directional: we applied cosine similarity to the
eigenvectors in both methods, and the result was either one or negative
one --- therefore the eigenvectors are the same, but for some the
direction is opposite. Such a difference was visible in images as well:
images generated through the same but opposite eigenvalue have inverted
pixels (where it is dark it becomes light and vice versa).

While utilizing either one of $S_1$ or $S_2$ produces the same (or
similar, as discussed earlier) result, there is a difference in
performance. It took *7.77* seconds to calculate the $S_1$ and just
*1.01* seconds to calculate $S_2$. Furthermore, while the calculation of
$S_1$ utilized *108.54MB* of memory, it took just *11.49 MB* to
calculate $S_2$. This is because the calculation of $S_1$ involves
multiplication of matrices along dimension, while $S_2$ involves
multiplication along classes. Since the number of features far exceeds
the number of classes, the calculation of $S_2$ is much faster.

## Application of Eigenfaces

After the eigenvectors are obtained, an image can be reconstructed back
under a different number of its largest components. This can be done by
projecting the obtained eigenvectors onto the mean image. We utilized
the alternate covariance matrix to calculate the eigenvectors, and the
results under the different number of eigenvectors can be found in Fig.
3, Fig. 4, and Fig. 5.

![A sample reconstructed identity from train
batch]({"reconstruct_1"}.png){#fig:reconstruct_1 width="8cm"}

![A sample reconstructed identity from test
batch]({"reconstruct_2"}.png){#fig:reconstruct_2 width="8cm"}

![A sample reconstructed identity from train
batch]({"reconstruct_3"}.png){#fig:reconstruct_2 width="8cm"}

The error has been calculated using the following principle, considering
*X* is the original image component and $X_R$ is the reconstructed image
component:
$$Err_{Reconst} = \frac{1}{N}\sum\limits_{i=1}^{N}(\lvert X_i - X_{Ri}\rvert / X_i)$$
We can notice that the reconstruction with very few bases (i.e. 5)
produces a generic image similar to the mean, and as the number of
eigenvalues is increased, the original identity starts to appear. Using
around 100 components yields unclear face features, while using several
components much larger than the rank of *S* seems to add some noise to
the image. It can be noticed that utilizing some components equal to the
rank of *S* yields zero reconstruction error. Since every eigenvector
varies along some axis, every one of them contributes to different
features of a face, and this matches the result of reconstructed images.

We performed the PCA-based face recognition using every possible number
of eigenvectors, from 1 to 500 to see which yields the best result. We
made use of the K-nearest neighbors classifier, a supervised learning
method. The 'ball-tree' algorithm yielded the best accuracy. As a
result, it was found that utilizing 308 eigenvectors produced the best
accuracy $(0.663)$. The corresponding confusion matrix and sample
success and failure cases can be found in Fig. 10 and Fig. 6,
respectively.

![Example predictions of PCA. Failed prediction of class 0 as 9 (on the
left), successful prediction of class 5 (center), and successful
prediction of class 10 (on the
right).]({"predictions"}.png){#fig:PCA_success_failure width="5.5cm"}

# INCREMENTAL PCA

Previously, all the train data had been used simultaneously for PCA, and
it is called Batch PCA. However, it is not efficient in terms of space
allocation to stack all data together, and sometimes it may not be
impossible at all. Incremental PCA allows the division of the train data
into batches and updates the PCA components incrementally. Data has been
divided into 4 subsets, each having 104 images. We performed face
recognition with Batch-PCA, Incremental PCA, and PCA based on only the
first batch with K-nearest Neighbours. The results can be found in
Table. 1.

For the sake of consistency, all models used the same number of
components (104). It can be noticed that both Incremental and Batch PCA
give the same accuracy. This means that the IPCA conserves the essential
features after every incremental update of the PCA model.

# LDA ENSEMBLE FOR FACE RECOGNITION

## PCA-LDA with NN Classifier

Both LDA and PCA are transformative techniques, where LDA aims to find
feature subspace to optimize class separability. However, applying LDA
directly can be problematic, since usually the number of features (2576)
far exceeds the number of classes (52). Since PCA reduces the number of
features, stacking them consequently helps with class discrimination.

PCA-LDA-based face recognition classifier has been implemented with a
K-nearest Neighbours classifier, to find the combination of PCA and LDA
parameters that would maximize the accuracy. Fig. 7 visualizes every
possible combination in $50 <= M_{PCA} <= 200$ and $1<= M_{LDA} <= 51$
that was tested. The best combination for PCA and LDA parameters are *55
and 15*, respectively. Then, for the given parameters, we calculated the
rank of within-class and between-class matrices, which are 55 and 51,
respectively. Fig.8 demonstrates the success and failure cases.

![Example predictions of PCA-LDA. Successful prediction of class 0 (on
the left) and failed prediction of class 17 as class 19 (on the
right).]({"PCA_LDA_prediction"}.png){#fig:PCA_LDA_3d width="5.5cm"}

## PCA-LDA Ensemble with NN Classifier

The same PCA-LDA with K-nearest Neighbours classifier can be done with
the ensemble as well, where we perform the task using T different models
working on different parts of the train data. We performed experiments
on 4 different kinds of ensemble combinations. Obtained confusion
matrices for bagging and feature space randomization (FSR), no bagging
and FSR, bagging and no FSR, and no bagging and no FSR, can be found in
Fig. 12, Fig. 13, Fig. 14, and Fig. 15, respectively. We set the number
of ensembles to be 10 and fixed the randomization in bagging to be 0.9.
All experiments used majority voting. The ensemble accuracies and error
metrics can be found in Table. 2.

The ensemble models demonstrate varying degrees of performance, with
notable differences in error metrics and recognition accuracy. When
bagging was employed, the committee machine error (0.0237) was
significantly lower than the average error of individual models
(0.0940). This suggests that bagging effectively reduces variance,
leading to a more robust and generalized model. In contrast, without
bagging, the average error of individual models (0.0213) was lower than
the committee machine error (0.0300), indicating a potential overfitting
issue or a lack of diversity among the models.

# GENERATIVE AND DISCRIMINATIVE SUBSPACE LEARNING

We had already discussed the PCA's ability to reconstruct data and LDA's
ability to extract discriminative features. The subspace learning that
would include both would need to have an objective function that would
balance them, and give the power to control the weight and preference.
Here, we explain the math behind such an objective function.

As discussed earlier, the objective function of PCA is to achieve the
maximum variance in data. Assuming that the direction of space is
defined by $w\in R^D$, the mean is $\mu = w^T \bar{x}$, which is a mean
of projections of every data point $x_n$ onto $w$. By definition,
variance ($\delta^2$) = $\frac{1}{N} \sum\limits_{n=1}^{N} (x - \mu)^2$.
Thus, the variance becomes
$$\delta^2 = \frac{1}{N} \sum\limits_{n=1}^{N} (w^T x_n -w^T \bar{x})^2 = \frac{1}{N} \sum\limits_{n=1}^{N} (w^T(x_n - \bar{x}))^2$$
By expanding and putting $w$ and $w^T$ out of the sum, we get
$$\delta^2 = w^T (\frac{1}{N} \sum\limits_{n=1}^{N} (x_n - \bar{x}) (x_n - \bar{x})^T) w$$
where
$\frac{1}{N} \sum\limits_{n=1}^{N} (x_n - \bar{x}) (x_n - \bar{x})^T$ is
a covariance matrix *S*. Thus, our objective function becomes
$$J_{PCA}(w) = w^T S w$$

The objective function for LDA is the Fisher's criterion, known as
generalized Rayleigh quotient:
$$J_{LDA}(w) = \frac{w^T S_B w}{w^T S_W w}$$ where *$S_B$* and $S_W$ are
between-class and within-class scatter matrices, respectively. Then, we
can combine the objective functions using a weighted sum approach:
$$\underset{w}{max} \hspace{0.1cm} J(w) = \alpha J_{LDA}(w) + (1 - \alpha) J_{PCA}(w) =$$
$$=\alpha \frac{w^T S_B w}{w^T S_W w} + (1 - \alpha) w^TSw$$ The
corresponding Lagrangian for this function will be
$$\mathcal{L}(w, \lambda) = \alpha w^TS_Bw + \lambda(\alpha(k - w^TS_Ww)) + (1-\alpha)w^TSw$$
Setting $\frac{\partial\mathcal{L}}{\partial w} = 0$ and simplifying the
expression, we get
$$\frac{\partial\mathcal{L}}{\partial w} = \alpha S_Bw - \lambda\alpha S_Ww + (1-\alpha)Sw = 0$$

# RF CLASSIFIER

Hyperparameter tuning in machine learning, particularly in Random Forest
classifiers, is a critical process to enhance model performance. We
consistently set the `random_state` parameter to 42 across all
experiments for direct and fair comparisons between different
hyperparameter configurations. Specifically, we examined the following
variations:

-   Number of Trees (`n_estimators`): 70, 100, 150, 200, and 250.

-   Maximum Tree Depth (`max_depth`): 5, 10, and 20.

-   Minimum Samples for Split (`min_samples_split`): 2, 5, and 10.

-   Minimum Samples per Leaf (`min_samples_leaf`): 1, 3, and 5.

## Hyperparameter Tuning

In our comprehensive exploration of hyperparameter tuning for the Random
Forest (RF) classifier, we observed a range of accuracies, varying from
approximately 54.81% to 86.54%, with the best configuration of
`'n_estimators': 250`, `'max_depth': 20`, `'min_samples_split': 2`, and
`'min_samples_leaf': 1`.

## Accuracy and Time Efficiency

The relationship between hyperparameters and model performance is
multifaceted. Increasing the number of trees generally improves accuracy
but with diminishing returns, and excessive numbers can lead to
overfitting. Tree depth is a critical factor: deeper trees can model
complex patterns but risk overfitting, while shallower trees may
underfit. The parameters `min_samples_split` and `min_samples_leaf`
control the tree's growth and complexity. Lower values in these
parameters allow for more complex models, but they increase the risk of
overfitting. In terms of time efficiency, more trees lead to longer
training times, and deeper trees take more time to train and test. The
parameters `min_samples_split` and `min_samples_leaf` have a varied
impact on training time, with higher values generally leading to faster
training due to simpler tree structures. The visualizations in Fig. 17
provide valuable insights into how each hyperparameter affects the
model's recognition accuracy and time efficiency.

## Confusion Matrix and Classification Report

The optimized setup demonstrated superior performance in terms of
overall recognition accuracy (observed in Table. 3), adeptly balancing
the model's complexity with its ability to generalize. Fig. 8
demonstrates some of the successful predictions. However, the detailed
classification report revealed that while the model excelled in most
classes, there were notable exceptions. The confusion matrix in Fig. 16
suggests that certain classes, such as 18 and 29, showed a significant
drop in precision (0.00), indicating potential issues with the model's
ability to generalize across all classes. Fig. 9 demonstrates classes 18
and 29 incorrect predictions. This inconsistency, despite the balanced
nature of the dataset, suggests that the model might be overfitting to
specific training data patterns or struggling with the inherent
complexity of certain classes.

::: center
::: {#table_example}
                  precision   recall   f1-score   support
  -------------- ----------- -------- ---------- ---------
     accuracy        \-         \-       0.87       104
    macro avg       0.89       0.87      0.85       104
   weighted avg     0.89       0.87      0.85       104

  : RF Classification Report
:::
:::

::: center
  -- -- -- ------- -- -- -- ------- -- -- -- ------- -- -- -- --
            \(a\)            \(b\)            \(c\)           
  -- -- -- ------- -- -- -- ------- -- -- -- ------- -- -- -- --
:::

![ Examples of successful predictions with RF. (a) Actual: 1,
Predicted: 1. (b) Actual: 2, Predicted: 2. (c) Actual: 3, Predicted:
3.]({"Figure5success"}.png){#fig:mean_face width="7cm"}

::: center
  -- -- -- ------- -- -- -- ------- -- -- -- ------- -- -- -- --
            \(a\)            \(b\)            \(c\)           
  -- -- -- ------- -- -- -- ------- -- -- -- ------- -- -- -- --
:::

![Examples of incorrect predictions with RF. (a) Actual: 18, Predicted:
3. (b) Actual: 29, Predicted: 24. (c) Example of an image with label 24.
In both cases, class 29 is incorrectly predicted as class
24.]({"task5failure"}.png){#fig:mean_face width="7cm"}

Comparing Batch PCA image reconstruction and an RF classifier applied to
face multiclass recognition, the sparsity observed in the PCA-based
reconstruction's confusion matrix (Fig. 10) suggests a less consistent
classification across multiple classes, which is further evidenced by an
overall accuracy of approximately 66%. This lower accuracy could be
indicative of PCA's reduced capability to capture the complex,
high-dimensional features necessary for robust face recognition. PCA's
linear nature may struggle with the nonlinear variations present in
facial data, leading to a more dispersed confusion matrix with errors
scattered across multiple classes. Conversely, the RF classifier
exhibits a 'neater' confusion matrix (Fig. 16), suggesting a more
concentrated diagonal indicative of correct classifications. The
substantially higher accuracy of 87% reflects the RF classifier's
effectiveness in face recognition tasks, likely due to its ensemble
approach that captures a broader range of data characteristics through
multiple decision trees. This approach not only improves generalization
over unseen data but also reduces the risk of overfitting, which is a
common challenge in complex classification tasks like face recognition.
The RF classifier's robustness to variations and noise in the data,
alongside its ability to model complex, nonlinear decision boundaries,
makes it a more suitable choice for the intricacies of multiclass face
recognition.

# APPENDIX {#appendix .unnumbered}

![Confusion matrix of batch
PCA]({"task1_confusion_matrix"}.png){#fig:conf_matrix_1 width="9cm"}

![3D Scatter plot for PCA-LDA
Accuracy]({"deterministic_pca_cropped"}.png){#fig:PCA_LDA_3d
width="8cm"}

![Confusion matrix of ensemble model with bagging and feature space
randomization]({"bagging"}.png){#fig:yes_bag_yes_FSR width="9cm"}

![Confusion matrix of ensemble model with no bagging and feature space
randomization]({"no_bagging"}.png){#fig:no_bag_yes_FSR width="9cm"}

![Confusion matrix of ensemble model with bagging and no feature space
randomization]({"bagging_no_feat"}.png){#fig:yes_bag_no_FSR width="9cm"}

![Confusion matrix of ensemble model with no bagging and no feature
space randomization]({"no_bagging_no_feat"}.png){#fig:no_bag_no_FSR
width="9cm"}

![Confusion matrix of optimized
RF]({"task5_confusion_matrix"}.png){#fig:mean_face width="9cm"}

![ Each hyperparameter affects the model's recognition accuracy and time
efficiency differently. Rows 1 to 3 represent accuracy, training time,
and testing time, respectively. Columns 1 to 3 represent the number of
trees, tree depth, and minimum samples split,
respectively.]({"time_efficiency_and_accuracy"}.png){#fig:mean_face
width="9cm"}

::: thebibliography
99

Turk, Matthew A. and Alex Paul Pentland. "Face recognition using
eigenfaces." *Proceedings. 1991 IEEE Computer Society Conference on
Computer Vision and Pattern Recognition* (1991): 586-591.
:::

[^1]: $^{1}$Department of Aerospace Engineering, KAIST,
    `kamran.qasimoff at kaist.ac.kr`

[^2]: $^{2}$Department of Aerospace Engineering, KAIST,
    `alexandra.suchshenko at kaist.ac.kr`
