---
layout: post
title: "k-Nearest Neighbors classifier, Naïve Bayes classifier in Data Mining "
author: "Sudip Bhujel"
categories: journal
tags: [datamining, datascience, machinelearning]
image: knnandnaivebayes.png
toc:
  sidebar: left
---

## I. Introduction

Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. The machine learning is classified into different categories viz. supervised machine learning, unsupervised learning, semi-supervised learning, and reinforcement machine learning. The supervised learning algorithm takes features as input, maps to a mapping function and approximates a result. The goal is to approximate the mapping function so well that when it gets new input that it can predict the output variables for that data. The supervised learning algorithm aims to find the pattern of the features to a particular result.

Classification, which is the task of assigning objects to one of several predefined categories, is a pervasive problem that encompasses many diverse applications. Examples include detecting spam email messages based upon the message header and content, categorizing cells as malignant or benign based upon the results of MRI scans, and classifying galaxies based upon their shapes.

The classification problems like email is spam or not, tumor is benign or malignant, etc. are binary classification as it deals with two categories in the target class. When there are more than two categories in the target class, the classification problem resides to multilabel classification and example might be like classifying cars company based on image whether it is Honda or Volkswagen or Renault.

The classification algorithm k-Nearest Neighbors classifier and Naïve Bayes classifier are two classifiers that better suits the classification problem. The performance metrics like Confusion matrix, Accuracy, F1 score, Precision, Recall, Heatmap, etc. gives the insight of model performance.

## II. Algorithms

The convention used in the derivation includes a collection of labeled examples
$$ \{(x*i,yi)\}*{i=1}^N $$
, where N is the size of the collection, 
$$x_i$$ is the D-dimensional feature vector of example
$$ i=1, 2, …, N $$ , $$y_i$$ is a real-valued target and every feature
$$ x_i^{(j)} $$
,
$$ j=1, 2, …, D $$
, is also a real number.

### A. k-Nearest Neighbors Classifier

k-Nearest Neighbors (kNN) is non parametric and instance-based learning algorithm. Contrary to other learning algorithms, it keeps all training data in memory. Once new, previously unseen example comes in, the kNN algorithm finds k training examples closest to x and returns the majority label.

The closeness of two examples is given by a distance function. For example, Euclidean distance is frequently used in practice. Euclidean distance between
$$ x_i $$
 and 
$$ x_k $$
is given as,

$$
d(\boldsymbol {x_i, x_k}) = \sqrt{(x_i^{(1)}-x_k^{(1)})^2 + (x_i^{(2)}-x_k^{(2)})^2 + ... + (x_i^{(N)}-x_k^{(N)})^2} \tag1
$$

The Euclidean distance in summation of the vector is given as;

$$
d(\boldsymbol {x_i, x_k}) = \sqrt{\sum_{j=1}^{D}(x_i^{(j)}-x_k^{(j)})^2} \tag2
$$

Another popular choice of the distance function is the negative cosine similarity. Cosine similarity defined as,

$$
s(\boldsymbol {x_i, x_k})=\frac{\sum_{j=1}^{D}x_i^{(j)}x_k^{(j)}}{\sqrt{\sum_{j=1}^{D}(x_i^{(j)})^2}\sqrt{\sum_{j=1}^{D}(x_k^{(j)})^2}} \tag3
$$

The Equation (3) gives a measure of similarity of the
directions of the two vectors and $$ 𝑠(\boldsymbol {x_i, x_k}) $$ can also be denoted
as $$ cos(\boldsymbol{x_i, x_k})$$
). If the angle between two vectors is $$ 0 $$ degrees,
then two vectors point to the same direction, and cosine
similarity is equal to $$ 1 $$. If the vectors are orthogonal, the cosine
similarity is $$ 0 $$. For vectors pointing in opposite directions, the
cosine similarity is $$ −1 $$. If we want to use cosine similarity as a
distance metric, we need to multiply it by $$ −1 $$. Other popular
distance metrics include Minkowski distance, Chebychev
distance, Mahalanobis distance, and Hamming distance. The
choice of the distance metric, as well as the value for k, are the
choices the analyst makes before running the algorithm.
The k-NN classifier starts with loading the data into memory.
The value of k (number of neighbors) defines the prediction
boundaries that means how much sorted distances are taken into
account to find the mode of the k labels. The algorithm takes
votes to classify the labels among selected k-neighbors. It
returns the majority class labels leaving behind minority. The
flowchart of the k-NN classifier is;

{% include figure.html path="assets/img/datamining-KNN.png" style="height: 50rem;" caption="Fig. 1.1: k-NN flowchart" alt="datamining-knn" %}

The selection of the hyperparameter k has a significant effect
on the classifier. In general, for the lower value of k, the
classifier may overfit on new unseen data. The value of k is
chosen such that balances bias and variance. When k is small,
we are restraining the region of a given prediction and forcing
our classifier to be “more blind” to the overall distribution. A
small value for K provides the most flexible fit, which will have
low bias but high variance. Graphically, our decision boundary
will be more jagged. On the other hand, a higher k averages
more voters in each prediction and hence is more resilient to
outliers. Larger values of k will have smoother decision
boundaries which means lower variance but increased bias.
The value of k is chosen such that the desired accuracy of kNN classifier is achieved. The simple method to calculate the
value of k is plotting error versus k graph and choosing the k on
which error is minimum.

### B. Naïve Bayes Classifier

Bayes’ Rule or Bayes’ Theorem is a statistical principle for
combining prior knowledge of the classes with new evidence
gathered from data. The class-conditional probability 𝑃(𝑋|𝑌),
and the evidence, P(X):The Bayes’ Rule (also known as the
Bayes’ Theorem) stipulates that:

$$
P(𝑌|\boldsymbol{X}) = \frac{P(\boldsymbol{X}|𝑌) P(𝑌)}
{P(\boldsymbol{X})}
\tag4
$$

In Bayes’ rule (4), it finds the probability of event 𝑌, given
that the event 𝑋 is true. Event 𝑋 is also termed as evidence.
𝑃(𝑌) is the priori of 𝑌 (the prior probability, i.e. Probability of
event before evidence is seen). 𝑃(𝑌|𝑿) is a posteriori
probability of 𝑋, i.e. probability of event after evidence is seen.
A Naïve Bayes classifier estimates the class-conditional
probability by assuming that the attributes are conditionally
independent, given the class label 𝑦. Here, 𝑃(𝑿) is a class
probability and 𝑃(𝑿|𝑦) is a conditional probability. The
conditional independence assumption can be formally stated as
follows:

$$
𝑃(\boldsymbol{X}|𝑌 = 𝑦) = \prod_{i=1}^d𝑃(𝑋_𝑖|𝑌 = 𝑦)\tag5
$$

Where each attribute set $$ \boldsymbol{X} = \{𝑋*1
,𝑋_2, … ,𝑋*𝑑\} $$ consists of d
attributes.
The Naïve Bayes is also called Simple Bayes as it assumes
that features of a measurement are independent of each other
and makes equal contribution to the outcome.

## III. METRICS

The classifier model doesn’t always give the accurate result.
There are some parameters to measure how the classifier
behave with unseen data to classify like Confusion matrix,
Accuracy, F1 score, Precision, Recall, Heatmap etc. The
different evaluation metrics are used for different kinds of
problems. We build a model, get feedback from metrics, make
improvements and continue until we achieve a desirable
accuracy. Evaluation metrics explain the performance of a
model. An important aspect of evaluation metrics is their
capability to discriminate among model results.

### A. Confusion Matrix

The confusion matrix is a table that summarizes how
successful the classification model is at predicting examples
belonging to various classes. One axis of the confusion matrix
is the label that the model predicted, and the other axis is the
actual label. In a binary classification problem, there are two
classes. Let’s say, the model predicts two classes: “spam” and
“not_spam”:

$$
\begin{array} {|r|r|}\hline  & & spam (predicted) & not_spam(predicted) \\ \hline & spam (actual)& 23 (TP) & 1 (FN) \\ \hline & not_spam (actual) & 12 (FP) & 556(TN) \\ \hline  \end{array}
$$

The above confusion matrix shows that of the 24 examples
that actually were spam, the model correctly classified 23 as
spam. In this case, we say that we have 23 true positives or TP
= 23. The model incorrectly classified 1 example as not_spam.
In this case, we have 1 false negative, or FN = 1. Similarly, of
568 examples that actually were not spam, 556 were correctly
classified (556 true negatives or TN = 556), and 12 were
incorrectly classified (12 false positives, FP = 12).

### B. Precision/Recall

The two most frequently used metrics to assess the model are
precision and recall. Precision is the ratio of correct positive
predictions to the overall number of positive predictions:

$$
precision = \frac{𝑇𝑃}{𝑇𝑃 + 𝐹𝑃}\tag6
$$

Recall is the ratio of correct predictions to the overall number
of positive examples in the datasets:

$$
recall = \frac{𝑇𝑃}{TP+FN} \tag7
$$

In the case of the spam detection problem, we want to have
high precision (we want to avoid making mistakes by detecting
that a legitimate message is spam) and we are ready to tolerate
lower recall (we tolerate some spam messages in our inbox).
The goal of classifier model is to choose between a high
precision or a high recall. It’s usually impossible to have both.
The hyperparameter tuning helps to maximize precision or
recall.

### C. Accuracy

Accuracy is given by the number of correctly classified
examples divided by the total number of classified examples. In
terms of the confusion matrix, it is given by:

$$
accuracy = \frac{TP+TN}{TP+TN+FP+FN} \tag8
$$

Accuracy is a useful metric when errors in predicting all
classes are equally important.

### D. F1 Score

F1-Score is the harmonic mean of precision and recall values
for a classification problem. The formula for F1-Score is as
follows:

$$
F1 = \frac{recall^{-1}+precision^{-1}}{2} \tag9
$$

$$
F1 = 2.\frac{precision.recall}{precision + recall} \tag{10}
$$

The general formula for positive real β, where β is chosen
such that recall is considered β times as important as precision,
is:

$$
F_{\beta}=(1+{\beta}^2) \cdot \frac{precision \cdot recall}{({\beta}^2 \cdot precision)+recall} \tag{11}
$$

The equation (11) or $$ 𝐹_𝛽 $$ measures the effectiveness of a model with respect
to a user who attaches β times as much importance to recall as precision.

### E. Heat Map

The heat map can be elucidated as a cross table or spreadsheet
which contains colors instead of numbers. The default color
gradient sets the lowest value in the heat map to dark blue, the
highest value to a bright red, and mid-range values to light gray,
with a corresponding transition (or gradient) between these
extremes. Heat maps are well-suited for visualizing large
amounts of multi-dimensional data and can be used to identify
clusters of rows with similar values, as these are displayed as
areas of similar color.

## IV. RESULT

The value of hyperparameter like k in the k-NN classifier
plays a significant role to correctly classify the labels or target
variables. The error versus k values plot provides a guideline to
choose k and the value of k with minimum error is chosen.

{% include figure.html path="assets/img/k_value_vs_error.png" class="img-fluid" caption="Fig. 5.1: Error versus K-value" %}

Fig. 5. 1 shows the fluctuation of error at different values of
k and the graph is not continuous. We would rather prefer to
calculate minimum error k-value than maximum error k-value
as minimum error k-value gives more accurate prediction. The
minimum error of k-NN classifier model for test set is at 𝑘 =
12 and the error is 0.0467 (i.e. 4.67%). Hence, 𝑘 = 12 is chosen
as k-value for k-NN classifier. The performance metrics of kNN classifier with parameters metric as ‘minkowski’,
neighbors as ‘12’are:

Confusion matrix: [[136 6] <br>
$\qquad$ $\qquad$ $\qquad$ [8 150]],<br>
Precision for label ‘0’ prediction: 0.94, <br>
Precision for label ‘1’ prediction: 0.96, <br>
Recall for label ‘0’ prediction: 0.96, <br>
Recall for label ‘1’ prediction: 0.95, <br>
F1-score for label ‘0’ prediction: 142, <br>
F1-score for label ‘1’ prediction: 158, <br>
Accuracy: 0.95 <br>
The model has classified the labels with 𝑇𝑃 = 136,𝐹𝑁 =
6, 𝐹𝑃 = 8, 𝑇𝑁 = 150 that means model misclassified 6 labels
as label ‘1’ which is actually label ‘0’ and misclassified 8 labels as label ‘0’ which is actually label ‘1’. Hence, the model has an accuracy of about 95%.

{% include figure.html path="assets/img/heat_map.jpg" class="img-fluid" caption="Fig. 5.2: Heat map predicted label over the true label" %}

Fig. 5. 2 Heat map predicted label over the true label
Heat map is a graphical representation of value in the
confusion matrix obtained from the predicted label and actual
target name. In the above heatmap, the red square denotes the
maximum value on the confusion matrix and with a decrease in
value the color fades up. Diagonal elements have a higher value
as shown in the heatmap which shows a higher performance of
the classification model and informs predicated label matches
the true label for any given input.

For the given model, “prime minister of nepal” supplied as
input assign a label “talk.politics.mideast” similarly , when
“joker” is supplied as input assign a label
“comp.sys.ibm.pc.hardware”. Here for two different input two
different label has been assigned out of which one label
assigned for the input “prime minister of nepal” is correct
whereas for “joker” correct label has not been assigned
properly which is due to naïve base treating the input as
independent values as well as lack of data being supplied.

## Conclusion

The two popular classifiers k-NN and Naïve Bayes provide
good accuracy to the model. Many parameters contribute to
model performance. The right choice of hyperparameter also
yields a better result. There is no rule of thumb to select the right
value of hyperparameter for the first trial and the
hyperparameter value that works fine for one model may not
yield the same result for another model. The good model is that
which considers all the performance metric parameters like
Accuracy, F1-score, Precision, Recall, etc. Though we have so
many metrics parameters to evaluate the model performance,
some analytics is needed to better explain the metric that
addresses classification problems in the best possible way. The
contribution of all performance metrics needs to be analyzed to
make the model accurate.

## Appendix

### A. k-NN classifier

```python
# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,
confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Importing the dataset
df = pd.read_csv('datasets/Dataset_1.csv', index_col=0)

# Quick look at data
df.head(4)

# Standardizing the variables
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET
CLASS', axis=1))
df_feat = pd.DataFrame(scaled_features,
columns=df.columns[:-1])

# After standardization
print(df_feat.head())

# train test split
X_train, X_test, y_train, y_test = train_test_split(
 scaled_features, df['TARGET CLASS'], test_size=0.3,
random_state=42)

# Initializing error and k_value list
error = []
k_value = []
for k in range(40):
    k_value.append(k+1)
    # Using KNN
    knn = KNeighborsClassifier(n_neighbors=k+1)
    print(knn.fit(X_train, y_train))
    pred = knn.predict(X_test)
    error_ = 1 - accuracy_score(y_test, pred)
    error.append(error_)

# Plotting k_value and error
plt.plot(k_value, error)
plt.xlabel('K value', fontsize=13)
plt.ylabel('Error', fontsize=13)
plt.savefig('k_value_vs_erro.png', dpi=1000,
bbox_inches="tight")
plt.show()

def performance_report(X_train, y_train, X_test, y_test,
n_neighbors=3):
    # k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    print(knn.fit(X_train, y_train))
    pred = knn.predict(X_test)
    confusion_matrix_ = confusion_matrix(y_test, pred)
    classification_report_ = classification_report(y_test, pred)
    return {
    'confusion_matrix': confusion_matrix_,
    'classification_report': classification_report_
 }

# to numpy
error_np = np.array(error)
k_value_np = np.array(k_value)
error_min_index = error_np.argmin().item() # numpy int to
python int
k_value_ = k_value_np[error_min_index]
print('K= {} and error= {}'.format(k_value_,
error_np[k_value_]))

# for minimum error
performance_report_ = performance_report(X_train,
y_train, X_test, y_test, n_neighbors=k_value_)

print('For k = {}: \n {}{}'.format(k_value_,
performance_report_['confusion_matrix'],
performance_report_['classification_report']))
```

### B. Naïve Bayes classifier

```python
# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets impor
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# importing the dataset
data = fetch_20newsgroups()
print(data.target_names)

# training the data on these categories
categories = data.target_names
train = fetch_20newsgroups(subset='train',
categories=categories)
test = fetch_20newsgroups(subset='test',
categories=categories)
print(train.data[5])

# Pipelining the model
model = make_pipeline(TfidfVectorizer(),
MultinomialNB())

# Fitting the data
model.fit(train.data, train.target)
labels = model.predict(test.data)

# heatmap
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d',
cbar=False, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
# plt.tight_layout()
plt.savefig('images/lab04/heat_map.jpg', dpi=1000,
bbox_inches="tight")
plt.show()

# predicting
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

# prediction
print(predict_category('Jesus christ'))
print(predict_category('Prime minister of Nepal'))
print(predict_category('Everest'))
```

## References

- P. Tan, M. Steinbach, V. Kumar and A. Karpatne, Introduction to Data
  Mining, Global Edition. Harlow, United Kingdom: Pearson Education
  Limited, 2019.
- A. Burkov, The hundred-page machine learning book, Global Edition.
  Quebec City, Canada, 2019.
