# Kaggle Competition: Restaurant Type Prediction
## Introduction
Welcome to the "Restaurant Type Prediction" Kaggle competition! In this competition, the goal is to develop data mining models that can accurately predict the type of restaurants based on the provided dataset, which contains details about restaurants and their reviews. This README will provide you with all the necessary information to get started with the competition.

## Dataset Overview
The dataset for this competition consists of the following key features:

### Restaurant details: 
Various attributes describing each restaurant, such as location, average cost for two, and more.
### Restaurant reviews: 
Textual reviews provided by customers for each restaurant.
### Restaurant type: 
The target variable, representing the type of restaurant (e.g., Fast Food, Italian, Chinese, etc.).

## Model and Approach
Task in this competition is to design data mining models that can accurately predict the type of a restaurant based on the provided features. I  primarily work with two machine learning techniques: Word2Vec for text data representation and a Multi-Layer Perceptron (MLP) for classification.
### To achieve a competitive solution for this competition, the following approach was used:

#### Data Preprocessing: 
Data cleaning and preprocessing were performed to handle missing values, encode categorical variables, and prepare the data for modeling.

#### Word2Vec for Text Data: 
Word2Vec, a word embedding technique, was applied to the textual restaurant reviews. This technique transforms words into dense vector representations, allowing the model to understand the semantic meaning of words in the reviews.

#### Multi-Layer Perceptron (MLP): 
An MLP neural network was trained on the engineered feature set to classify the restaurant types. The architecture of the MLP was optimized through experimentation.

#### Model Evaluation: 
The model's performance was assessed using appropriate evaluation metrics, with a focus on accuracy.

## Achieved Accuracy
After training and fine-tuning the models, an accuracy of 83.19% was achieved on the test dataset. This performance demonstrates the effectiveness of the approach in predicting restaurant types accurately.

## Conclusion
This Kaggle competition provides an exciting opportunity to apply data mining techniques to predict restaurant types based on a rich dataset of restaurant details and customer reviews.