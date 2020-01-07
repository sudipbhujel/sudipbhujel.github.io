---
layout: post
title: "Data Mining Parameters"
author: "Sudip Bhujel"
categories: journal
tags: [datamining, EDA]
image: data-mining-parameters.jpeg
---

Datamining covers everything that are related with the data from collection of raw data to EDA and preparation of input to AI algorithm. We have lots of parameters for describing the data. Some of them we are going to discuss are Impurity index, Central of tendency, Eigenvalue/ Eigenvector, PCA in Classification. <br>
 <strong> Abstract: <i> The impurities measurement parameter of dataset
like Entropy, Gini, Classification Error aims to find the error while classifying the labels. The attribute with less value of impurity will be chosen out of attribute contenders. The measure
of central tendency like mean, median, quartiles, etc. along with boxplot gives the idea about the distribution of data and outliers which leads then how to treat the data to get the most information
out of it. The features/ attributes are important parameters for any machine learning algorithm, large-sized attributes result in a more accurate prediction which means that the model has high
accuracy. The computational cost for a model with a large number of attributes is generally high. The best model is that which takes as least attributes as possible without losing the information and has reasonable accuracy. Principal Component Analysis (PCA) is a feature extraction method that uses orthogonal linear projections to capture the underlying variance of the data. It reduces the number of least wanted features for prediction without losing the overall information of data. </i></strong>

## Entropy
Entropy is a measure of impurity, disorder or uncertainty in a bunch of examples i.e. it is an indicator of how messy our data is. In Decision Trees, the goal is to tidy the data. Entropy controls how a Decision Tree decides to split the data. It affects how a Decision Tree draws its boundaries so that the outcomes from the algorithm will have purely classified objects.

$$E(x) = \sum_{x\epsilon X} p(x)log_2 p(x)
$$

Where,
S = The current dataset for which entropy is being calculated <br>
X = Set of classes in S<br>
p(x) = The probability of each set S

![Impurity vs probability](/assets/img/Impurity_vs_Probability.png)
<p style="color:grey; text-align:center; font-style:italic"> Impurity Index versus Probability, Impurity Indices are Entropy, Gini, and Classification Error</p>

## Gini
Impurity measures such as entropy and Gini index tend to favor attributes that have a large number of distinct values . If we consider the same example as in entropy, the gini index is computed using the following equation:

$$
G(S) = 1-\sum_{x\epsilon X} |p(x)|^2 
$$

Where,
S = The current dataset for which entropy is being calculated <br>
X = Set of classes in S <br>
p(x) = The probability of each set S

## Classification Error
Classification error is a measure of impurity at a node and defined for classification error at a node t as,

$$ Error(t) = 1 − maxP(i|t) $$ 

The classification error made by node ranges minimum 0 when all records belong to one class to maximum $$ (1 − 1/n_c ) $$ when records are equally distributed among all classes.


## Covariance Matrix
Variance measures the variation of a single random variable
(like the height of a person in a population), whereas covariance
is a measure of how much two random variables vary together
(like the height of a person and the weight of a person in a
population). The covariance matrix can be calculated using
covariance, which is a square matrix given by C I,j = σ(x i , x j )
where C ∈ R d xd and d describe dimension or number of
random variables of the data (e.g. the number of features like
height, width, weight, etc.). The calculation for the covariance
matrix can be also expressed as:

$$
C = \frac{1}{n-1} \sum_{i=1} ^n (X_i-\overline{X} )(X_i-\overline{X} )^T
$$

The covariance matrix for two dimensions is given by,

$$
\begin{pmatrix} \sigma(x,x) & \sigma(x,y) \\\ \sigma(y,x) & \sigma(y,y) \end{pmatrix}
$$

The covariance matrix is symmetric since $$ \sigma(x_i, x_j) = \sigma(x_j, x_i) $$.

## Eigenvalue and Eigenvector
In linear algebra, an eigenvector of a linear transformation is
a nonzero vector that changes at most by a scalar factor when
that linear transformation is applied to it. The corresponding
eigenvalue is the factor by which the eigenvector is scaled. For
linear equations:

$$
Av = λv
$$

In this equation A is an n-by-n matrix, v is a non-zero n-by-1
vector and $$ \lambda $$ is a scalar (which may be either real or complex).
Any value of $$ \lambda $$ for which this equation has a solution is known
as eigenvalue of the matrix A. It is sometimes also called the
characteristics value. The vector, v, which corresponds to this
value is called an eigenvector. The eigen problem can be written
as

$$
A. v − \lambda. v = 0 \\
A. v − \lambda. I. v = 0 \\
(A − \lambda. I). v = 0 
$$

If v is non-zero, this equation will only have a solution if
$$ |A − \lambda. I| = 0 $$
This equation is called the characteristic equation of A, and is
an nth order polynomial in $$\lambda$$ with n roots. These roots are called
the eigenvalues of A. We will only deal with the case of n
distinct roots, though they may be repeated. For each
eigenvalue, there will be an eigenvector for which the
eigenvalue equation is true.

## Distances
<strong>Euclidean distance</strong> is a measure of the distance between two
points in Euclidean space. Mathematically,

$$
dist = \sqrt{\sum_{k=1}^n (p_k - q_k)^2}
$$

Where n is the number of dimensions (attributes) and $$p_k$$ and
$$q_k$$ are, respectively, the $$k^th$$ attributes (components) or data
objects p and q.
<strong>Minkowski Distance </strong> is a generalization of Euclidean
distance and given as,

$$
dist = \left(\sum_{k=1}^n |p_k - q_k|^r \right)^{\frac{1}{r}}
$$

Where r is a parameter, n is the number of dimensions
(attributes) and $$p_k$$ and $$q_k$$ are, respectively, the k th attributes
(components) or data objects p and q.
- r = 1, it becomes Manhattan distance.
- r = 2, it becomes Euclidean distance.
- $$r \to \infty $$, it becomes supremum distance.

## Similarity
The similarity is the measure of how much alike two data
objects are. The similarity in a data mining context is usually
described as a distance with dimensions representing features
of the objects. A small distance indicating a high degree of
similarity and a large distance indicating a low degree of
similarity. The similarity is subjective and is highly dependent
on the domain and application.<br>
<strong>Cosine Similarity</strong> of two document vectors is given as,

$$
cos(d_1, d_2) = \frac{d_1 . d_2}{||d_1||.||d_2||}
$$

Where ||d|| is the length of vector d.<br>
Cosine similarity is for comparing two real-valued vectors,
but <strong>Jaccard similarity</strong> is for comparing two binary vectors
(sets). Mathematically,

$$
J_g (a,b) = frac{sum_i min(a_i, b_i)}{sum_i max(a_i, b_i)}
$$

For example, $$ t_1 = (1, 1,0,1), t_2 = (2,0,1,1)$$, the generalized
Jaccard similarity index can be computed as follows:

$$
J(t_1, t_2) = \frac{1+0+0+1}{2+1+1+1} = 0.4
$$

## PCA
Principal Component Analysis (PCA) is a feature extraction
method that uses orthogonal linear projections to capture the
underlying variance of the data. The main idea of principal
component analysis (PCA) is to reduce the dimensionality of a
data set consisting of many variables correlated with each other,
either heavily or lightly, while retaining the variation present in
the dataset, up to the maximum extent. It reduces the dimension
of the data with the aim of retaining as much information as
possible. In other words, this method combines highly
correlated variables to form a smaller number of an artificial set
of variables which is called “principal components” that
account for the most variance in the data.

## CONCLUSION
The measure of central of tendency, similarity, etc. are the
part of Exploratory Data Analysis (EDA). The EDA itself
doesn’t give the model for prediction but extremely useful for
getting the sense of information from data. This gives an idea
about how to get started with the data. Impurity indices like
Entropy, Gini, and Classification Error in the classification
helps examine how classification algorithm struggles to classify
the items based on their attributes. The impurity index helps
find the depth of the decision tree algorithm.

## References
- [P. Tan, M. Steinbach, V. Kumar and A. Karpatne, Introduction to Data
Mining, Global Edition. Harlow, United Kingdom: Pearson Education
Limited, 2019.](https://www.amazon.com/Introduction-Mining-Whats-Computer-Science/dp/0133128903/ref=sr_1_5?keywords=data+mining&qid=1577535801&sr=8-5)