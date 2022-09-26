# Home Credit Default Risk

The main objective of this project was to train a model which would be able to predict the capability of a client to return a loan. In particular, I coded the model to return either 1 or 0, being this the values for being able or not, rather than the probability of returning it. The dataset I used was one from a Kaggle competition, you can see it on the Main Notebook of the project.

## Motivation

Nowadays, it is estimated that around 3800 millions of people have a bank account. Imagine that just if 1% of them ask for a loan, he would be talking about 38 million of citizens. Having a model which predicts wether if a client will be able to return the loan is such a powerfull tool... 

The other main motivation to start this project was the amount of things I was going to be able to learn and put into practice on such an interesting and challenging project.

## Table of Contents


**[1. Credit Fraud Analysis](#heading--1)**

  * [1.1. Exploratory Data Analysis](#heading--1-1)
  * [1.2. Handilng Imbalanced Data](#heading--1-2)
    * [1.2.1. UnderSampling](#heading--2-1-1)
    * [1.2.2. OverSampling](#heading--2-1-1)
  * [1.3. Dimensionallity Reduction](#heading--1-2)
    * [1.3.1. PCA](#heading--2-1-1)
    * [1.3.2. TSNE](#heading--2-1-1)
  *  [1.4. Preprocesing and Feature Engineering](#heading--1-2)
     * [1.4.1. Correct Outilers/Anomslous values](#heading--2-1-1)
     * [1.4.2. Impute values for Missing Data](#heading--2-1-1)
     * [1.4.3. Encode Categorical Features](#heading--2-1-1)
     * [1.4.4. Feature Scaling](#heading--2-1-1)
  *  [1.5. Training Models](#heading--1-2)
  *  [1.6. Models Evaluation](#heading--1-2)
     * [1.6.1. RandomSearch](#heading--2-1-1)
* [1.7. Scikit-Learn PipeLines](#heading--2-1-1)

## Technologies and Teachings

On this project I dealt with a very big dataset, I had more than 120 features from more than 300k clients of the bank. This obligate to oneself, to start thinking about the feature and model selection more carefully. In order to mantain a "Computantionally Efficient" project, as this was done locally on my laptop. 

I really focus on doing a wide and insightful EDA. Also, I learned the problems and techniques to deal with a very common issue, handling a dataset with Imbalanced data.
Even more, I had to implement Dimensionallity Reduction techniques, in order to be able of visualize the separabillity of the classes.

Later on, I coded myself functions to complete the preprocessing step. This had a didactic end, as it can be done much more easily and quickly with the Scikit-Learn PipeLines as it is showm on the last part of the project. 

Finally, as in every ML project I tried several different Classification Models and compare their efficiency, in order to choose one. The evaluation of the models were done with tha Confussion Matrix and the Roc-Auc Value. It can be seen graphically.

One last comment, there are 3 separate python files which are imported during the project. This are the preprocessing, visualization and evalution files. Each one of them contains functions whith the purpose mention on their titles. This was done to mantain the aesthetics of the project, and logically, to avoid repetition.

Hope you enjoy and learn as much as myself with this project!

