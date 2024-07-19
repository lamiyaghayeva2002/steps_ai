from rank_bm25 import BM25Okapi
import numpy as np
from nltk.corpus import wordnet
import nltk
from chunking import model  # Import the model from chunking.py

nltk.download('wordnet')

def expand_query(query):
    expanded_query = []
    for word in query.split():
        synonyms = wordnet.synsets(word)
        expanded_query.extend([lemma.name() for syn in synonyms for lemma in syn.lemmas()])
    return list(set(expanded_query))

def retrieve_and_rerank(query, topic_chunks, collection):
    expanded_query = expand_query(query)

    corpus = [" ".join(item) for item in topic_chunks.values()]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = expanded_query
    bm25_scores = bm25.get_scores(tokenized_query)
    initial_candidates = [corpus[i] for i in np.argsort(bm25_scores)[::-1][:10]]

    query_embedding = model.encode(query).tolist()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], anns_field="embedding", param=search_params, limit=10)

    reranked_results = []
    for result in results[0]:
        similarity_score = np.dot(query_embedding, result.entity.embedding)
        reranked_results.append((result, similarity_score))

    reranked_results.sort(key=lambda x: x[1], reverse=True)
    return reranked_results
