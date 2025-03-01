# %% [markdown]
# Random Forest Classifier

# %%
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import nltk
from math import log2

# Download NLTK data for stemming
nltk.download('punkt')

# Load Data
df1 = pd.read_csv(r'D:\Fall Semester 2024\CS 438\Model 1\scraping_with_EDA\scraping.ipynb\cleaned_combined_articles.csv')

print("Data loaded successfully.")
print(f"Columns: {df1.columns.tolist()}")
print(f"Number of samples: {len(df1)}")

# Encode Labels
print("Encoding labels...")
le = LabelEncoder()
df1['gold_label'] = le.fit_transform(df1['gold_label'])
print(f"Unique labels after encoding: {list(le.classes_)}")

# Apply Stemming
print("Applying stemming to the text data...")
stemmer = PorterStemmer()
df1['cleaned_content'] = df1['cleaned_content'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Convert Text Data to Features
print("Converting text data into numerical features using TfidfVectorizer...")
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df1['cleaned_content']).toarray()
y = df1['gold_label'].values
print(f"Feature matrix shape: {X.shape}")

# Combine Features and Labels for Custom Implementation
dataset = np.hstack((X, y.reshape(-1, 1)))
print("Dataset prepared by combining features and labels.")

# Split data into train, validation, and test sets
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)
print(f"Training samples: {len(train)}, Validation samples: {len(val)}, Testing samples: {len(test)}")

# Define Entropy Impurity
def entropy_impurity(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    entropy = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        group_labels = [row[-1] for row in group]
        for class_val in classes:
            p = group_labels.count(class_val) / size
            if p > 0:
                score -= p * log2(p)
        entropy += (score * (size / n_instances))
    return entropy

# Split Dataset
def split_dataset(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Get Best Split Using Entropy
def get_best_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    features = np.random.choice(range(len(dataset[0]) - 1), n_features, replace=False)
    best_index, best_value, best_score, best_groups = 999, 999, float('inf'), None
    for index in features:
        for row in dataset:
            groups = split_dataset(index, row[index], dataset)
            entropy = entropy_impurity(groups, class_values)
            if entropy < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], entropy, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

# Create Terminal Node
def create_terminal_node(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Recursive Splitting
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = create_terminal_node(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = create_terminal_node(left), create_terminal_node(right)
        return
    if len(left) <= min_size:
        node['left'] = create_terminal_node(left)
    else:
        node['left'] = get_best_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    if len(right) <= min_size:
        node['right'] = create_terminal_node(right)
    else:
        node['right'] = get_best_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)

# Build Tree
def build_tree(train, max_depth, min_size, n_features):
    print("Building a decision tree...")
    root = get_best_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    print("Tree built successfully.")
    return root

# Predict
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Random Forest Implementation
class RandomForest:
    def __init__(self, n_trees, max_depth, min_size, n_features):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.trees = []

    def subsample(self, dataset, ratio):
        n_sample = round(len(dataset) * ratio)
        return [dataset[i] for i in np.random.choice(len(dataset), n_sample, replace=True)]

    def fit(self, train):
        print(f"Training Random Forest with {self.n_trees} trees...")
        for i in range(self.n_trees):
            print(f"Training tree {i + 1}/{self.n_trees}...")
            sample = self.subsample(train, 1.0)
            tree = build_tree(sample, self.max_depth, self.min_size, self.n_features)
            self.trees.append(tree)
        print("Random Forest training completed.")

    def predict(self, row):
        predictions = [predict(tree, row) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

    def predict_dataset(self, test):
        print("Making predictions on the dataset...")
        return [self.predict(row) for row in test]

# Hyperparameters
n_trees = 20  # Number of trees in the forest
max_depth = 30  # Maximum depth of a tree
min_size = 1  # Minimum samples per leaf node
n_features = int(np.sqrt(X.shape[1]))  # Number of features to consider at each split

# Initialize and Train the Random Forest
rf = RandomForest(n_trees=n_trees, max_depth=max_depth, min_size=min_size, n_features=n_features)
rf.fit(train)



# %% [markdown]
# Evaluation Metrics

# %%
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Validate the model
print("\nEvaluating on Validation Data...")
val_predictions = rf.predict_dataset(val)
val_labels = val[:, -1]

val_accuracy = np.mean(val_predictions == val_labels)
val_precision = precision_score(val_labels, val_predictions, average='weighted', zero_division=0)
val_recall = recall_score(val_labels, val_predictions, average='weighted', zero_division=0)
val_f1 = f1_score(val_labels, val_predictions, average='weighted', zero_division=0)
val_conf_matrix = confusion_matrix(val_labels, val_predictions)

print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Validation Precision: {val_precision:.2f}")
print(f"Validation Recall: {val_recall:.2f}")
print(f"Validation F1-Score: {val_f1:.2f}")
print("\nValidation Confusion Matrix:")
print(val_conf_matrix)

# Test the model
print("\nEvaluating on Test Data...")
test_predictions = rf.predict_dataset(test)
test_labels = test[:, -1]

test_accuracy = np.mean(test_predictions == test_labels)
test_precision = precision_score(test_labels, test_predictions, average='weighted', zero_division=0)
test_recall = recall_score(test_labels, test_predictions, average='weighted', zero_division=0)
test_f1 = f1_score(test_labels, test_predictions, average='weighted', zero_division=0)
test_conf_matrix = confusion_matrix(test_labels, test_predictions)

print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Precision: {test_precision:.2f}")
print(f"Test Recall: {test_recall:.2f}")
print(f"Test F1-Score: {test_f1:.2f}")
print("\nTest Confusion Matrix:")
print(test_conf_matrix)

# Print Detailed Classification Report
print("\nClassification Report on Test Data:")
print(classification_report(test_labels, test_predictions, target_names=le.classes_))


