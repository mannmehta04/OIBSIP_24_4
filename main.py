import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import nltk
from nltk.corpus import stopwords
import string
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
import mplcursors

df = pd.read_csv('./data/spam.csv', encoding='ISO-8859-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'text']

nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['text'] = df['text'].apply(preprocess_text)

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

# Confusion matrix
fig, ax = plt.subplots()
plot_confusion_matrix(model, X_test_vec, y_test, display_labels=['ham', 'spam'], cmap=plt.cm.Blues, normalize='true', ax=ax)
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
mplcursors.cursor(hover=True).connect("add")
plt.show()

# ROC curve
fig, ax = plt.subplots()
plot_roc_curve(model, X_test_vec, y_test, ax=ax)
plt.title('ROC Curve')
mplcursors.cursor(hover=True).connect("add")
plt.show()

# Precision-recall curve
fig, ax = plt.subplots()
plot_precision_recall_curve(model, X_test_vec, y_test, ax=ax)
plt.title('Precision-Recall Curve')
mplcursors.cursor(hover=True).connect("add")
plt.show()

# Histogram of Text Lengths
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df, x='text_length', hue='label', bins=50, kde=True)
plt.title('Histogram of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
mplcursors.cursor(hover=True).connect("add")
plt.show()

# Word Clouds
spam_text = ' '.join(df[df['label'] == 'spam']['text'])
ham_text = ' '.join(df[df['label'] == 'ham']['text'])

plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
wordcloud_spam = WordCloud(width=800, height=800, background_color='white').generate(spam_text)
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.title('Word Cloud for Spam Messages')
plt.axis('off')

plt.subplot(1, 2, 2)
wordcloud_ham = WordCloud(width=800, height=800, background_color='white').generate(ham_text)
plt.imshow(wordcloud_ham, interpolation='bilinear')
plt.title('Word Cloud for Ham Messages')
plt.axis('off')
mplcursors.cursor(hover=True).connect("add")
plt.show()

# Bar Chart of Most Frequent Words
def plot_most_frequent_words(text, label, n=10):
    words = text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[word[0] for word in most_common_words], y=[word[1] for word in most_common_words])
    plt.title(f'Most Frequent Words in {label} Messages')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    mplcursors.cursor(hover=True).connect("add")
    plt.show()

plot_most_frequent_words(spam_text, 'Spam')
plot_most_frequent_words(ham_text, 'Ham')

# Box Plot of TF-IDF Scores
tfidf_scores_spam = X_train_vec[y_train == 'spam'].toarray().mean(axis=0)
tfidf_scores_ham = X_train_vec[y_train == 'ham'].toarray().mean(axis=0)

plt.figure(figsize=(10, 6))
sns.boxplot(x=['spam']*len(tfidf_scores_spam) + ['ham']*len(tfidf_scores_ham), y=list(tfidf_scores_spam) + list(tfidf_scores_ham))
plt.title('Box Plot of TF-IDF Scores')
plt.xlabel('Label')
plt.ylabel('TF-IDF Score')
mplcursors.cursor(hover=True).connect("add")
plt.show()
