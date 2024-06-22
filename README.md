

---

# Crypto Currency Detection Using Machine Learning

## Introduction

This project aims to detect whether a given set of data represents cryptocurrency or not using various machine learning techniques. We will employ data preprocessing, feature engineering, and model training using the LightGBM algorithm. The dataset used for this project comes from a hackathon and includes features related to system performance and network activity.

## Data Loading and Exploration

First, we load the datasets and explore their structure to understand the available features and target variable.

```python
# Load files
train = pd.read_csv('/content/drive/MyDrive/indabax-rwanda-2024-hacktathon20240614-2435-ccan8g/Train.csv')
test = pd.read_csv('/content/drive/MyDrive/indabax-rwanda-2024-hacktathon20240614-2435-ccan8g/Test.csv')
ss = pd.read_csv('/content/drive/MyDrive/indabax-rwanda-2024-hacktathon20240614-2435-ccan8g/SampleSubmission.csv')

# Preview datasets
print("Train Dataset Preview:")
print(train.head())

print("\nTest Dataset Preview:")
print(test.head())

print("\nSample Submission File Preview:")
print(ss.head())
```

## Data Analysis and Preprocessing

### Exploring Target Distribution

```python
# Distribution of target variable
train.Label.value_counts()
```

### Statistical Summary

```python
# Train dataset statistical summary
train.describe(include='all')
```

### Checking for Missing Values and Duplicates

```python
# Check for missing values
train.isnull().sum().any(), test.isnull().sum().any()

# Check for duplicates
train.duplicated().any(), test.duplicated().any()
```

### Visualizing Target Variable Distribution

```python
# Visualize the distribution of the target variable
sns.set_style('darkgrid')
plt.figure(figsize=(12, 6))
sns.countplot(x='Label', data=train)
plt.title('Target Variable Distribution')
plt.show()
```

### Correlation Heatmap

```python
# Correlation heatmap of numeric columns
numeric_cols = train.select_dtypes(include=np.number)
corr = numeric_cols.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr, cmap='RdYlGn', annot=True, center=0)
plt.title('Correlogram', fontsize=15, color='darkgreen')
plt.show()
```

### Feature Engineering

We perform feature engineering to create new features that might be useful for our model.

```python
# Separate target variable from features
target = train.Label
train = train.drop("Label", axis=1)

# Combine train and test datasets for feature engineering
data = pd.concat([train, test], ignore_index=True)

# Define a function to rank the data
def rank_4_3_2_1(x):
    no_outliers = x[(x - x.mean()).abs() <= 3 * x.std()]
    q1 = no_outliers.quantile(0.25)
    q2 = no_outliers.quantile(0.5)
    q3 = no_outliers.quantile(0.75)
    ranks = pd.Series(index=x.index)
    ranks[x >= q3] = 2
    ranks[(x >= q2) & (x < q3)] = 3
    ranks[(x >= q1) & (x < q2)] = 4
    ranks[x < q1] = 1
    return ranks

# Rank and create new features
columns = data.drop("ID", axis=1).columns.tolist()
to_remove = []

for col in columns:
    ranks = rank_4_3_2_1(data[col])
    data[f"{col}_rank"] = ranks
    to_remove.append(f"{col}_rank")

# Sum of the ranks
rank_sums = data.filter(regex="_rank$").sum(axis=1)
data["Rank Sum"] = rank_sums
data = data.drop(to_remove, axis=1)

# Additional feature engineering
data["Speed of Operations to Speed of Data Bytes"] = data["I/O Data Operations"] / data[" I/O Data Bytes"]
data["Time for a single Process"] = data["Time on processor"] / data["Number of subprocesses"]
data["Ratio of data flow"] = data["Received Bytes (HTTP)"] / data["Bytes Sent/sent"]
data["Ratio of Packet flow"] = data["Network packets received"] / data["Network packets sent"]
data["Total Page Errors"] = data["Time on processor"] * data["Page Errors/sec"]
data["Network Usage"] = data["Bytes Sent/sent"] + data["Received Bytes (HTTP)"]
data["Network Activity Rate"] = data["Network packets sent"] + data["Network packets received"]
data["Page Fault Rate"] = (data["Pages Read/sec"] + data["Pages Input/sec"]) / data["Page Errors/sec"]
data["Network Latency"] = data["Network packets sent"] - data["Network packets received"]
data["Disk Latency"] = (data["Disk Reading/sec"] + data["Disc Writing/sec"]) / data["I/O Data Operations"]

# Drop less useful columns
data = data.drop(["Disc Writing/sec", "Pages Input/sec", "ID"], axis=1)

# Split the combined data back into train and test sets
train = data.iloc[:train.shape[0],]
test = data.iloc[train.shape[0]:,]
y = target
```

## Model Training and Evaluation

We use LightGBM for model training and evaluation. Stratified K-Fold cross-validation is used to ensure that each fold has the same proportion of target classes.

```python
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

# Define parameters for the LightGBM model
params = {
    'learning_rate': 0.185243125886494,
    'subsample': 0.2128564969905326,
    'colsample_bytree': 0.5046224462041669,
    'max_depth': 13,
    'scale_pos_weight': 4,
    'n_estimators': 995
}

# Initialize StratifiedKFold
fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Lists to store results
predictions = []
scores = []

# Cross-validation
for train_index, test_index in fold.split(train, y):
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    classifier = LGBMClassifier(**params)
    classifier.fit(X_train, Y_train)
    preds = classifier.predict(X_test)
    score = f1_score(Y_test, preds)
    scores.append(score)
    predictions.append(classifier.predict(test))
    print("F1 Score: ", score)

# Print the average F1 score
print("Average F1 Score: ", np.mean(scores))

# Feature importance
feature_importance_df = pd.DataFrame(classifier.feature_importances_, columns=['importance'])
feature_importance_df['feature'] = X_train.columns

plt.figure(figsize=(20, 12))
sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by=['importance'], ascending=False).head(50))
plt.title('LGBMClassifier Feature Importance (Top 50)')
plt.show()
```

## Predictions and Submission

We use the trained model to make predictions on the test set and prepare the submission file.

```python
# Prepare the submission file
sub = ss.copy()
predictions = pd.DataFrame(predictions)
predictions = predictions.mode().iloc[0]
sub["Target"] = predictions
sub["ID"] = test["ID"]
sub.to_csv("indaba_24_crypto.csv", index=False)
```

## Conclusion

In this project, we successfully applied LightGBM to predict whether a given dataset represents cryptocurrency. We performed thorough data preprocessing and feature engineering to enhance model performance. The model was evaluated using Stratified K-Fold cross-validation, and the results were saved in the required format for submission.

## Dependencies

Here is a list of the dependencies used in this project:

```plaintext
pandas
numpy
seaborn
matplotlib
scikit-learn
lightgbm
xgboost
catboost
```

To save the dependencies to a requirements file:

```python
!pip freeze > requirements.txt
from google.colab import files
files.download("requirements.txt")
```

---


