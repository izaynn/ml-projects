# 🌿 Iris Species Classification

**A comprehensive machine learning project that classifies Iris flowers into Setosa, Versicolor, and Virginica using Logistic Regression, K-Nearest Neighbors, and Decision Tree classifiers, with full exploratory data analysis and model comparison.**

## 📂 Dataset Source
The dataset is the classic **Iris Dataset** originally introduced by Ronald Fisher in 1936. 

- **UCI Machine Learning Repository:** [https://archive.ics.uci.edu/dataset/53/iris](https://archive.ics.uci.edu/dataset/53/iris)
- **Kaggle:** [https://www.kaggle.com/datasets/uciml/iris](https://www.kaggle.com/datasets/uciml/iris)

The dataset contains 150 samples with 4 features: `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, and `PetalWidthCm`. The `Id` column was dropped during preprocessing as it contains no predictive information.

## 🤖 Algorithms Used
This project implements and compares the following classification algorithms:

- **Logistic Regression** – A linear model for multiclass classification using a one-vs-rest strategy.
- **K-Nearest Neighbors (KNN)** – A distance-based algorithm tested with `k=5` neighbors.
- **Decision Tree Classifier** – A tree-based model with `max_depth=3` to prevent overfitting.

## 📊 Results

| Algorithm | Test Accuracy | CV Accuracy (5-Fold) |
| :--- | :--- | :--- |
| Logistic Regression | 0.9333 | 0.9600 |
| KNN (k=5) | 0.9333 | 0.9600 |
| Decision Tree | **0.9667** | **0.9733** |

The **Decision Tree** achieved the highest cross-validation accuracy, making it the best-performing model on this dataset.

## 📈 Visualizations
The notebook includes extensive EDA with boxplots, scatter plots, pair plots, and a correlation heatmap. Below is the pair plot showing feature relationships across species:

![Pairplot](images\pair_plot.png)

> *Pairplot of all Iris features, colored by species. Notice the clear separation of Setosa from the other two species.*


## 🚀 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/izaynn/ml-projects/tree/main/iris-species-classifier
   cd your-repo

## Conclusion
Based on the EDA, Petal Length and Petal Width emerged as the most important features, with Setosa being perfectly separable from the other two species. The correlation heatmap confirmed a strong relationship between petal length and petal width (r ≈ 0.96), making them somewhat redundant.

Among the three models, the Decision Tree with max_depth=3 achieved the best cross-validation accuracy (0.9733) and performed consistently well on the test set (0.9667). While KNN and Logistic Regression also performed well, the Decision Tree was chosen for its simplicity, interpretability, and strong performance. The tree's rules (printed in the notebook) show that it primarily splits on PetalLengthCm, which aligns with our EDA findings.

This project demonstrates the importance of EDA, feature selection, and cross-validation in building reliable classification models.