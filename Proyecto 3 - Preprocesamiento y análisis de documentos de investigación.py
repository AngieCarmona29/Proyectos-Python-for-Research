import os
import re
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def load_and_extract_text(directory_path):
    papers = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(directory_path, filename))
            papers[filename] = " ".join(page.extract_text() for page in reader.pages)
    return papers

def preprocess_and_tokenize(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r"[^\w\s]", "", text.lower())
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words and len(word) > 2]

def extract_sections(text):
    abstract = re.search(r'abstract(.*?)(introduction|background|1\.)', text, re.DOTALL | re.IGNORECASE)
    references = re.search(r'references(.*)', text, re.DOTALL | re.IGNORECASE)
    return (abstract.group(1).strip() if abstract else "Abstract not found",
            references.group(1).strip() if references else "References not found")

def visualize_frequent_terms(tokens, top_n=20):
    freq_dist = FreqDist(tokens)
    frequent_terms = freq_dist.most_common(top_n)

    WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(frequent_terms)).to_image().show()

    words, counts = zip(*frequent_terms)
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title("Top Frequent Terms")
    plt.show()

    return frequent_terms

def main():
    directory_path = "PDFs"
    papers = load_and_extract_text(directory_path)

    all_tokens, abstracts, references = [], [], []
    for filename, text in papers.items():
        print(f"\nProcesando archivo: {filename}")
        tokens = preprocess_and_tokenize(text)
        abstract, reference = extract_sections(text)
        all_tokens.extend(tokens)
        abstracts.append((filename, abstract))
        references.append((filename, reference))

        print(f"Resumen: {abstract[:200]}...\nReferencias: {reference[:200]}...")

    frequent_terms = visualize_frequent_terms(all_tokens)
    print("\nTérminos más frecuentes:")
    for term, freq in frequent_terms:
        print(f"{term}: {freq}")


    print("\nHallazgos del informe:")
    for filename, abstract in abstracts:
        print(f"Archivo: {filename}, Resumen: {abstract[:100]}...")
    for filename, reference in references:
        print(f"Archivo: {filename}, Referencias: {reference[:100]}...")

if __name__ == "__main__":
    main()