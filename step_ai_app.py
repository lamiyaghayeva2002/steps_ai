import json
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
import numpy as np
from transformers import pipeline
from rank_bm25 import BM25Okapi
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, IndexType, Collection
import streamlit as st
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

# Load the question-answering pipeline from Hugging Face
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Connect to MILVUS server
connections.connect(host='localhost', port='19530')

def chunk_data(data):
    chunked_data = []
    sentences = []
    for url, content in data.items():
        paragraphs = content.get('paragraphs', [])
        if not paragraphs:
            continue

        # Tokenize paragraphs into sentences
        for paragraph in paragraphs:
            sentences.extend(sent_tokenize(paragraph))

    # Encode sentences into embeddings
    sentence_embeddings = model.encode(sentences)

    # Use K-Means clustering to group sentences into chunks
    num_chunks = 30 # Limit to 5 chunks per document
    kmeans = KMeans(n_clusters=num_chunks, random_state=0)
    kmeans.fit(sentence_embeddings)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_embeddings)

    # Sort chunks by sentence order
    chunks = [sentences[idx] for idx in sorted(closest)]

    chunked_data.append({
        'url': url,
        'chunks': chunks
    })

    return chunked_data

def topic_modeling(data):
    # Preprocess data for topic modeling
    texts = []
    for item in data:
        texts.extend(item['chunks'])

    # Use LDA for topic modeling
    lda = LatentDirichletAllocation(n_components=10, random_state=0)
    doc_topic_dists = lda.fit_transform(abs(model.encode(texts)))  # Assuming model.encode returns vectors for each sentence

    topic_chunks = defaultdict(list)
    for i, topic_dist in enumerate(doc_topic_dists):
        topic_chunks[np.argmax(topic_dist)].append(texts[i])

    return topic_chunks

def encode_topic_chunks(topic_chunks):
    encoded_chunks = {}
    for topic, chunks in topic_chunks.items():
        # Encode chunks using the SentenceTransformer model
        chunk_embeddings = model.encode(chunks)
        encoded_chunks[topic] = {
            'chunks': chunks,
            'embeddings': chunk_embeddings.tolist()  # Convert embeddings to list for JSON serialization
        }
    return encoded_chunks

def create_index(collection, index_type=IndexType.IVF_FLAT, nlist=4096):
    # Create index schema
    index_params = {
      "metric_type": "L2",
      "index_type": "IVF_FLAT",
      "params": {"nlist": 1024}
    }
    index = collection.create_index(field_name="embedding", index_type=index_type, index_params=index_params)
    return index

def insert_vectors(collection, encoded_topic_chunks):
    vectors = []
    id_counter = 0
    for topic, data in encoded_topic_chunks.items():
        for embedding in data['embeddings']:
            vectors.append({"id": id_counter, "len": len(embedding), "embedding": embedding})
            id_counter += 1

    # Insert vectors into MILVUS collection
    collection.insert(vectors)
    return len(vectors)

def expand_query(query):
    expanded_query = []
    for word in query.split():
        synonyms = wordnet.synsets(word)
        expanded_query.extend([lemma.name() for syn in synonyms for lemma in syn.lemmas()])
    return list(set(expanded_query))

def retrieve_and_rerank(query, topic_chunks):
    # Step 1: Query expansion
    expanded_query = expand_query(query)

    # Step 2: Retrieve initial candidates using BM25
    corpus = [" ".join(item) for item in topic_chunks.values()]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = expanded_query
    bm25_scores = bm25.get_scores(tokenized_query)
    initial_candidates = [corpus[i] for i in np.argsort(bm25_scores)[::-1][:10]]  # Example: top 10 candidates

    # Step 3: Retrieve embeddings from MILVUS
    query_embedding = model.encode(query).tolist()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.query(
        expr="",
        output_fields=["id", "embedding"],
        anns_field="embedding",
        param=search_params,
        limit=10
    )

    # Step 4: Re-rank based on relevance and similarity
    reranked_results = []
    for result in results:
        result_embedding = result["embedding"]
        similarity_score = np.dot(query_embedding, result_embedding)
        reranked_results.append((result, similarity_score))

    reranked_results.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score

    return reranked_results

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit UI
st.title("Document Chunking, Topic Modeling, and Question Answering")

uploaded_file = st.file_uploader("Choose a JSON file", type="json")
if uploaded_file is not None:
    crawled_data = json.load(uploaded_file)

    # Step 1: Chunk data based on semantic similarity
    chunked_data = chunk_data(crawled_data)
    # Step 2: Topic modeling to organize chunks by topic
    topic_chunks = topic_modeling(chunked_data)
    
    # Insert vectors into MILVUS and create index
    encoded_data = encode_topic_chunks(topic_chunks)
    collection = Collection(name="milvus_storing")
    insert_vectors(collection, encoded_data)
    create_index(collection)

    st.write("Data has been processed and indexed.")

    query = st.text_input("Enter your query:")
    if query:
        results = retrieve_and_rerank(query, topic_chunks)
        for idx, (result, similarity_score) in enumerate(results, start=1):
            st.write(f"Rank {idx}: Similarity Score: {similarity_score:.4f}, Result ID: {result['id']}")

        context = " ".join([chunk for chunk in list(dict(topic_chunks).values())[0]])  # Assuming topic_chunks[0] contains relevant chunks
        question = st.text_input("Enter your question for the context:")
        if question:
            answer = answer_question(question, context)
            st.write(f"Question: {question}")
            st.write(f"Answer: {answer}")
