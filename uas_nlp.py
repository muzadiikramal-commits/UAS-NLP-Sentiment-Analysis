import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. MEMBACA DATASET DARI FILE EKSTERNAL ---
# Ini membuat tugasmu lebih profesional karena memisahkan data dan kode
df = pd.read_csv('data_komentar.csv')

# --- 2. PREPROCESSING TEKS ---
stemmer = StemmerFactory().create_stemmer()

def preprocess_teks(teks):
    teks = str(teks).lower() # Case Folding
    teks = re.sub(r'[^\w\s]', '', teks) # Cleaning
    teks = stemmer.stem(teks) # Stemming
    return teks

print("Sedang melakukan Preprocessing...")
df['ulasan_bersih'] = df['ulasan'].apply(preprocess_teks)

# --- 3. PEMBAGIAN DATASET (80:20) ---
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['ulasan_bersih'])
y = df['sentimen']

# Menggunakan train_test_split sesuai instruksi soal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. PELATIHAN MODEL ---
model = MultinomialNB()
model.fit(X_train, y_train)

# --- 5. EVALUASI ---
y_pred = model.predict(X_test)
print("\n=== LAPORAN EVALUASI ===")
print(classification_report(y_test, y_pred))