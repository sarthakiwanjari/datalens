# DataLens

**Upload any dataset. See what is wrong with it, fix it, and find out which algorithm learns it best.**

DataLens is an interactive web app that turns a raw CSV or Excel file into a complete data science workflow in minutes: data quality audit, one-click cleaning, exploratory analysis, accuracy of every algorithm, and a downloadable report.

**Live demo:** https://datalens-data-quality-and-ml-accuracy-sarthakiwanjari.streamlit.app/

## Features

| Tab | What it does |
|---|---|
| **Overview** | Data preview, column types, summary statistics, and a data quality score out of 100 |
| **Data quality** | Missing values per column, duplicate rows, and outliers found with the IQR rule |
| **Clean** | Remove duplicates, drop sparse columns, fill gaps with median, mean, KNN or mode, then download the cleaned CSV |
| **Explore** | Distributions, correlation heatmap and scatter plots |
| **Accuracy** | Accuracy of every algorithm in one place, with train, test and 5-fold cross-validation scores and an overfitting check |
| **Models** | Full metrics for each model: confusion matrix, actual vs predicted, feature importance |
| **Clustering** | Four clustering algorithms, an elbow curve and a PCA cluster plot |
| **Report** | Download the whole analysis as an HTML report or an Excel workbook |

## Algorithms

**Classification** (accuracy, precision, recall, F1)
Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM, Naive Bayes

**Regression** (R², MAE, RMSE)
Linear Regression, Ridge, Lasso, ElasticNet, KNN, Decision Tree, Random Forest, Gradient Boosting, SVR

**Clustering** (silhouette, Davies-Bouldin, Calinski-Harabasz, and purity when labels exist)
KMeans, Agglomerative, Gaussian Mixture, DBSCAN

The app also sweeps KNN across k = 1 to 25 and highlights the best value.

## How it works

1. The data quality score combines completeness, uniqueness and outlier share.
2. Models run inside scikit-learn pipelines. Imputation, scaling and one-hot encoding are fitted on training data only, which avoids data leakage.
3. The task type is detected from the target column and can be overridden.
4. Text columns are encoded automatically. Columns with more than 30 unique text values, such as timestamps, are left out.
5. Large datasets are sampled (20,000 rows for models, 5,000 for clustering) to keep the app responsive.

## Run locally

```bash
git clone https://github.com/sarthakiwanjari/datalens.git
cd datalens
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac or Linux
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. No dataset on hand? Pick one of the built-in samples in the sidebar.

## Project structure

```
datalens/
├── app.py                  # Streamlit app
├── requirements.txt        # Python dependencies
├── .streamlit/config.toml  # Theme and upload size
└── README.md
```

## Tech stack

Python, pandas, NumPy, scikit-learn, Plotly, Streamlit, openpyxl, matplotlib

## Limitations

- Models use default settings. Hyperparameter tuning is not built in yet.
- Very small datasets (under about 150 rows) give unstable accuracy scores.
- Uploads are limited to 50 MB.

## Roadmap

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] SHAP model explanations
- [ ] XGBoost and LightGBM
- [ ] Imbalanced data handling with SMOTE
- [ ] Predict on new data and download the trained model

## Author

**Sarthaki**, Data Science and AI/ML
[GitHub](https://github.com/sarthakiwanjari)