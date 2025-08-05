#logistic regression with ridge
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/Training_Essay_Data.csv')

label_counts = df['generated'].value_counts()
print("Label counts:\n", label_counts)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['generated'])  # 0: human, 1: AI

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['generated'],
    test_size=0.2,
    random_state=42,
    stratify=df['generated']
)


vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    class_weight='balanced',  
    max_iter=1000
)


clf.fit(X_train_vec, y_train)


y_pred = clf.predict(X_test_vec)


print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
