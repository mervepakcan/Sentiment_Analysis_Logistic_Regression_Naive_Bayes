# 🎬 IMDB Sentiment Analysis with Machine Learning

This project performs sentiment classification on IMDB movie reviews using two supervised machine learning algorithms: **Logistic Regression** and **Multinomial Naive Bayes**. The aim is to detect whether a given review expresses a positive or negative sentiment.

📑 **Table of Contents**
- [Project Overview](#project-overview)  
- [Data Description](#data-description)  
- [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)  
- [Modeling](#modeling)  
- [Results & Evaluation](#results--evaluation)  
- [Technologies](#technologies)  
 

---XX

## 🔍 Project Overview

The project focuses on:

- Classifying movie reviews as **positive** or **negative**
- Implementing Logistic Regression **from scratch**
- Comparing it with a traditional **Naive Bayes** model
- Applying standard **text preprocessing** techniques
- Evaluating model performance using **classification metrics**

---

## 📊 Data Description

The dataset used in this project is the Large Movie Review Dataset (IMDB) developed by Andrew L. Maas et al. (2011) at Stanford University. It contains 50,000 highly polar movie reviews—25,000 for training and 25,000 for testing, evenly split between positive and negative sentiments.

Format: CSV
Balanced classes: 50% positive, 50% negative
Language: English
Label: Binary (positive, negative)
🔗 Original Source:
Stanford Large Movie Review Dataset v1.0

📁 Dataset on Kaggle:
IMDB Dataset of 50K Movie Reviews

---

## 🧹 Preprocessing & Feature Engineering

Steps included:

- Lowercasing text  
- Removing punctuation and special characters  
- Tokenization (using NLTK)  
- Stopword removal  
- Lemmatization  
- TF-IDF feature extraction

---

## 🤖 Modeling

### 1. Logistic Regression (From Scratch)
- Implemented using pure Python (without scikit-learn)
- Trained using gradient descent
- Evaluated on a hold-out test set

### 2. Multinomial Naive Bayes
- Applied to the same preprocessed text features
- Used scikit-learn’s implementation
- Includes ratio analysis and custom word-level prediction

---

## 📈 Results & Evaluation

| Model               | Accuracy | Notes                            |
|--------------------|----------|----------------------------------|
| Logistic Regression | ~62%     | Implemented from scratch         |
| Naive Bayes         | ~84.7%   | Higher accuracy and performance  |

- **Naive Bayes** outperformed Logistic Regression in terms of accuracy.
- Logistic Regression was implemented manually using mathematical formulation without libraries.
- Naive Bayes was implemented from scratch using probabilistic word frequency calculations.

---

## 🛠️ Technologies

- Python  
- Jupyter Notebook  
- Pandas, NumPy  
- Scikit-learn  
- NLTK  
- Matplotlib, Seaborn
