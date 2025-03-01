# %% [markdown]
# Naive Bayes Classifier 

# %%
import pandas as pd
import numpy as np
df1 = pd.read_csv(r'D:\Fall Semester 2024\CS 438\Model 1\scraping_with_EDA\scraping.ipynb\cleaned_combined_articles.csv')

from sklearn.metrics import classification_report, confusion_matrix

texts = df1['cleaned_content']
labels = df1['gold_label']

# tokenization
def preprocess(text):
    return text.split()

processed_texts = texts.apply(preprocess)

#add each word to the vocabulary. 
vocabulary = set()
for text in processed_texts:
    for word in text:
        vocabulary.add(word)

# vocab_to_index = {word: i for i, word in enumerate(vocabulary)}
vocab_to_index = {}
index = 0
for word in vocabulary:
    vocab_to_index[word] = index
    index += 1

def encode_text(text):
    vector = np.zeros(len(vocabulary))
    for word in text:
        if word in vocab_to_index:
            vector[vocab_to_index[word]] += 1
    return vector

#storing the frequencies of each word in X array
X = np.array([encode_text(text) for text in processed_texts])
y = labels.values


def train_test_split(X, y, test_size=0.2, random_state=30):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_i = int(len(X) * (1 - test_size))
    train_indices= indices[:split_i]
    test_indices = indices[split_i:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # count the occurrences of each class in y
        class_counts = {}
        for label in y:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        #calculate class probabilities (P(class))
        class_probs = {}
        for c, count in class_counts.items():
            prob = count / n_samples
            class_probs[c] = prob
        self.class_probs = class_probs

        self.feature_probs = {c: np.zeros(n_features) for c in class_counts}
        for c in class_counts:
            X_c = X[y == c]  
            feature_sums = np.sum(X_c, axis=0) + 1  
            self.feature_probs[c] = feature_sums / feature_sums.sum()

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = {}
            for c in self.class_probs:
                log_prob = np.log(self.class_probs[c])
                log_prob += np.sum(np.log(self.feature_probs[c]) * x)
                class_scores[c] = log_prob
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)

nb_model = NaiveBayesClassifier()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)


# %% [markdown]
# Evaluation Metrics

# %%
def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)

print("Accuracy:", accuracy(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(y_test, y_pred))


