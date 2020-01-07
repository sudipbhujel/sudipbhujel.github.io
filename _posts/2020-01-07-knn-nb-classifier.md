---
layout: post
title: "k-Nearest Neighbors classifier, Naïve Bayes classifier in Data Mining "
author: "Sudip Bhujel"
categories: journal
tags: [datamining, datascience, machinelearning]
image: knnandnaivebayes.png
---

# Introduction
Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. The machine learning is classified into different categories viz. supervised machine learning, unsupervised learning, semi-supervised learning, and reinforcement machine learning. The supervised learning algorithm takes features as input, maps to a mapping function and approximates a result. The goal is to approximate the mapping function so well that when it gets new input that it can predict the output variables for that data. The supervised learning algorithm aims to find the pattern of the features to a particular result. 

Classification, which is the task of assigning objects to one of several predefined categories, is a pervasive problem that encompasses many diverse applications. Examples include detecting spam email messages based upon the message header and content, categorizing cells as malignant or benign based upon the results of MRI scans, and classifying galaxies based upon their shapes. 

The classification problems like email is spam or not, tumor is benign or malignant, etc. are binary classification as it deals with two categories in the target class. When there are more than two categories in the target class, the classification problem resides to multilabel classification and example might be like classifying cars company based on image whether it is Honda or Volkswagen or Renault.   

The classification algorithm k-Nearest Neighbors classifier and Naïve Bayes classifier are two classifiers that better suits the classification problem. The performance metrics like Confusion matrix, Accuracy, F1 score, Precision, Recall, Heatmap, etc. gives the insight of model performance.

## Algorithms
The convention used in the derivation includes a collection of labeled examples 
$$ \{(x_i,yi)\}_{i=1}^N $$
, where N is the size of the collection, 
$$x_i$$ is the D-dimensional feature vector of example 
$$ i=1, 2, …, N $$ , $$y_i$$ is a real-valued target and every feature 
$$ x_i^{(j)} $$ 
, 
$$ j=1, 2, …, D $$
, is also a real number. 

### k-Nearest Neighbors Classifier 

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