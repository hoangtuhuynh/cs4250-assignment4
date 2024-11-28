from pymongo import MongoClient
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# MongoDB connection setup
client = MongoClient('mongodb://localhost:27017/')
db = client['search_engine']

# Clear the collections if they already exist
db.terms.drop()
db.documents.drop()

# Documents from Question 3
documents = [
    "After the medication, headache and nausea were reported by the patient.",
    "The patient reported nausea and dizziness caused by the medication.",
    "Headache and dizziness are common effects of this medication.",
    "The medication caused a headache and nausea, but no dizziness was reported."
]

# Preprocessing: Lowercase, remove punctuation, and tokenize
def preprocess(doc):
    doc = re.sub(r'[^\w\s]', '', doc.lower())  # Remove punctuation and lowercase
    tokens = doc.split()
    return tokens

# Generate unigrams, bigrams, and trigrams
def generate_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Insert documents into the database
doc_collection = db.documents
doc_data = [{"_id": i+1, "content": doc} for i, doc in enumerate(documents)]
doc_collection.insert_many(doc_data)

# Build the inverted index
vocabulary = {}
term_collection = db.terms

for doc_id, content in enumerate(documents, start=1):
    tokens = preprocess(content)
    unigrams = tokens
    bigrams = generate_ngrams(tokens, 2)
    trigrams = generate_ngrams(tokens, 3)
    all_terms = unigrams + bigrams + trigrams
    
    for term in all_terms:
        if term not in vocabulary:
            vocabulary[term] = len(vocabulary) + 1  # Assign a unique position ID
        term_id = vocabulary[term]
        
        # Check if the term already exists in the database
        term_entry = term_collection.find_one({"_id": term_id})
        if term_entry:
            term_entry["docs"].append({"doc_id": doc_id, "tfidf": 0})  # Placeholder for TF-IDF
            term_collection.replace_one({"_id": term_id}, term_entry)
        else:
            term_collection.insert_one({
                "_id": term_id,
                "term": term,
                "pos": term_id,
                "docs": [{"doc_id": doc_id, "tfidf": 0}]
            })

# Compute TF-IDF for terms in documents
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True, stop_words=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Update TF-IDF values in the inverted index
for term, index in zip(feature_names, range(tfidf_matrix.shape[1])):
    tfidf_scores = tfidf_matrix[:, index].toarray().flatten()
    term_entry = term_collection.find_one({"term": term})
    if term_entry:
        for doc in term_entry["docs"]:
            doc["tfidf"] = float(tfidf_scores[doc["doc_id"] - 1])
        term_collection.replace_one({"_id": term_entry["_id"]}, term_entry)

# Query processing
queries = [
    "nausea and dizziness",
    "effects",
    "nausea was reported",
    "dizziness",
    "the medication"
]

# Function to compute cosine similarity
def cosine_similarity(query_vector, doc_vector):
    dot_product = np.dot(query_vector, doc_vector)
    magnitude = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
    if magnitude == 0:
        return 0.0
    return dot_product / magnitude

# Rank documents for each query
results = []
for query in queries:
    query_tokens = preprocess(query)
    query_vector = np.zeros(len(vocabulary))
    for token in query_tokens:
        if token in vocabulary:
            query_vector[vocabulary[token] - 1] = 1
    
    document_scores = []
    for doc_id, content in enumerate(documents, start=1):
        doc_vector = np.zeros(len(vocabulary))
        for term_entry in term_collection.find():
            for doc in term_entry["docs"]:
                if doc["doc_id"] == doc_id:
                    doc_vector[term_entry["pos"] - 1] = doc["tfidf"]
        
        score = cosine_similarity(query_vector, doc_vector)
        if score > 0:
            document_scores.append((content, score))
    
    # Sort documents by score in descending order
    document_scores = sorted(document_scores, key=lambda x: x[1], reverse=True)
    results.append((query, document_scores))

# Output results
output = []
for query, docs in results:
    output.append(f"Query: {query}")
    for doc, score in docs:
        output.append(f"\"{doc}\", {score:.2f}")
    output.append("")

# Save the results to a text file
output_path = "output.txt"
with open(output_path, "w") as file:
    file.write("\n".join(output))

output_path
