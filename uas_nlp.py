import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('data_komentar.csv')

# Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def cleaning_process(text):
    # Case folding dan penghapusan karakter khusus
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Proses stemming
    text = stemmer.stem(text)
    return text

print("Proses preprocessing data sedang berjalan...")
df['ulasan_bersih'] = df['ulasan'].apply(cleaning_process)

# Transformasi teks menggunakan TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['ulasan_bersih'])
y = df['sentimen']

# Split data: 80% training dan 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediksi dan evaluasi hasil
y_pred = model.predict(X_test)
print("\n=== Hasil Evaluasi Model ===")
print(classification_report(y_test, y_pred))