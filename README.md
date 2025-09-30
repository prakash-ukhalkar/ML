# Machine Learning using Python Programming for Beginners

| Active Development | License: MIT | Python |

## Introduction & Project Vision

Welcome to `Machine Learning (ML)`!

This repository serves as a beginner-friendly, step-by-step guide to mastering Machine Learning (ML) using the Python programming language. My approach is uniquely focused on **Practical Learning**, **Code Implementation**, and **Concept Understanding**, providing comprehensive insights through hands-on examples and real-world datasets.

Whether you're a student, a self-learner, or someone transitioning into data science, this repo provides a clear, structured path to understanding the fundamental concepts of ML.

### Focus Areas

• **Scikit-learn Mastery**: Deep-dive into ML algorithms like Linear Regression, Logistic Regression, Decision Trees, and Clustering.
• **Data Preprocessing**: Feature engineering, handling categorical variables, train-test splits, and data visualization.
• **Algorithm Implementation**: Step-by-step implementation of classic ML algorithms with detailed explanations.
• **Storytelling**: Every analysis is accompanied by clear, educational markdown explanations and practical business applications.

## Repository Structure

The project is organized as a sequential learning path via Jupyter Notebooks.

```
ML/
│
├── README.md                                    <- This file
├── 01_Simple Linear Regression_ML/
│   ├── 01_Simple_Linear_Regression_ML.ipynb    <- Basic linear regression concepts
│   ├── house_prices_inr.csv                    <- Sample dataset
│   ├── new_areas.csv                           <- Test data
│   ├── predicted_house_prices.csv             <- Model predictions
│   └── Exercise_SLR/                           <- Practice exercises
├── 02_Multiple_Linear_Regression_ML/
│   ├── 02_Multiple_Linear_Regression_ML.ipynb  <- Multiple feature regression
│   ├── car_prices.csv                          <- Sample dataset
│   └── Exercise_MLR/                           <- Practice exercises
├── 03_Dummy_Variables_and_One_Hot_Encoding_ML/
│   ├── 01_Dummy_Variable_and_One_Hot_Encoding.ipynb <- Categorical data handling
│   ├── car_prices_ohe.csv                      <- Sample dataset
│   └── Exercise_Dummy_Variables_OHE/           <- Practice exercises
├── 04_Training_and_Testing_Dataset_ML/
│   ├── 01_Training_and_Testing_Dataset.ipynb   <- Train-test split concepts
│   ├── house_prices_tt.csv                     <- Sample dataset
│   └── Exercise_Training_and_Testing_Dataset/  <- Practice exercises
├── 05_Simple_Logistic_Regression_ML/
│   ├── 01_Simple_Logistic_Regression.ipynb     <- Binary classification
│   ├── customer_churn.csv                      <- Sample dataset
│   └── Exercise_Logistic_Regression_ML/        <- Practice exercises
├── 06_Multiclass_Logistic_Regression_ML/
│   ├── 01_Multiclass_Logistic_Regression_ML.ipynb <- Multi-class classification
│   └── 01_Exercise_Multiclass_Logistic_Regression_ML/
├── 07_Feature_Engineering_ML/
│   ├── 01_Outlier_Detection_Removal_Using_Quantile_ML.ipynb
│   ├── 02_Outlier_Detection_Removal_Using_Z-Score_Std-Dev_ML.ipynb
│   ├── 03_Outlier_Detection_Removal_Using_IQR_ML.ipynb
│   └── 01_Exercise_Feature_Engineering_ML/
├── 08_Decision_Tree_Classification_ML/
│   ├── 01_Decision_Tree_Classification_ML.ipynb <- Tree-based classification
│   └── 01_Exercise_Decision_Tree_Classification_ML/
├── 09_Random_Forest_Classification_ML/
│   ├── 01_Random_Forest_Classification_ML.ipynb <- Ensemble methods
│   └── 01_Exercise_Random_Forest_Classification_ML/
├── 10_Support_Vector_Machines_ML/
│   ├── 01_Support_Vector_Machines_ML.ipynb     <- SVM classification
│   └── 01_Exercise_Support_Vector_Machine_ML/
├── 11_KFold_Cross_Validation_ML/
│   ├── 01_KFold_Cross_Validation_ML.ipynb      <- Model validation
│   └── 01_Exercise_KFold_Cross_Validation_ML/
├── 12_Naive_Bayes_Classification_ML/
│   ├── 01_GaussianNB_Classification_ML.ipynb   <- Probabilistic classification
│   └── Exercise/
├── 13_kNN_Classification_ML/                    <- k-Nearest Neighbors
├── 14_KMeans_Clustering_ML/                     <- Unsupervised learning
├── 15_GridSearchCV_Hyper_Parameter_Tuning_ML/   <- Model optimization
└── ML_Code/                                     <- Additional code examples
```

## Getting Started

To run the notebooks locally, follow these steps.

### 1. Prerequisites

• **Python**: Version 3.8 or higher.
• **Git**: For cloning the repository.

### 2. Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prakash-ukhalkar/PU_ML_Code_Git.git
   cd ML
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   # Using venv (standard Python)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   # OR
   jupyter lab
   ```

### 3. Running the Analysis

Start with the notebook `01_Simple_Linear_Regression_ML.ipynb` and proceed sequentially through the numbered directories.

## Notebooks: A Detailed Roadmap

| # | Topic | Description |
|---|-------|-------------|
| 01 | Simple Linear Regression | Understanding linear relationships, model training, predictions, and evaluation metrics (R², MAE, MSE, RMSE). |
| 02 | Multiple Linear Regression | Working with multiple features, feature selection, and interpreting coefficients in multi-dimensional space. |
| 03 | Dummy Variables & One Hot Encoding | Handling categorical data, avoiding dummy variable trap, and preparing data for ML algorithms. |
| 04 | Training and Testing Datasets | Data splitting strategies, model validation, and understanding overfitting vs. generalization. |
| 05 | Simple Logistic Regression | Binary classification, sigmoid function, probability prediction, and decision boundaries. |
| 06 | Multiclass Logistic Regression | Extending logistic regression to multiple classes, one-vs-rest, and softmax classification. |
| 07 | Feature Engineering | Outlier detection and removal using Quantile, Z-Score, and IQR methods for data quality improvement. |
| 08 | Decision Tree Classification | Tree-based learning, entropy, Gini impurity, and interpretable decision-making models. |
| 09 | Random Forest Classification | Ensemble methods, bagging, feature importance, and reducing overfitting through averaging. |
| 10 | Support Vector Machines | Margin maximization, kernel trick, and handling non-linearly separable data. |
| 11 | K-Fold Cross Validation | Robust model evaluation, cross-validation strategies, and performance estimation. |
| 12 | Naive Bayes Classification | Probabilistic classification, Bayes' theorem, and handling categorical and continuous features. |
| 13 | k-Nearest Neighbors | Instance-based learning, distance metrics, and non-parametric classification. |
| 14 | K-Means Clustering | Unsupervised learning, cluster analysis, and pattern discovery in unlabeled data. |
| 15 | GridSearchCV & Hyperparameter Tuning | Model optimization, parameter search, and achieving best performance through systematic tuning. |

## Dependencies

The core libraries used are:

• `pandas` - Data manipulation and analysis
• `numpy` - Numerical computing
• `scikit-learn` - Machine learning algorithms
• `matplotlib` - Data visualization
• `seaborn` - Statistical data visualization
• `jupyter` - Interactive notebook environment

## Who This Is For

• **Beginners** with basic Python knowledge.
• **Students** exploring AI and ML for the first time.
• **Self-learners** looking for structured, hands-on ML education.
• **Anyone curious** about how machines learn from data.

## Contributions

Contributions are welcome! If you'd like to improve examples, add topics, or fix something, feel free to open a pull request.

Happy Learning!

## Author

`ML` repo is created and maintained by **[Prakash Ukhalkar](https://github.com/prakash-ukhalkar)**

Built with ❤️ for the Python community