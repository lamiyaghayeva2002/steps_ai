import streamlit as st
import json
from chunking import chunk_data
from topic_modeling import topic_modeling
from vector_database import encode_topic_chunks, create_collection, create_index, insert_vectors
from retrieval import retrieve_and_rerank
from question_answering import answer_question
from pymilvus import Collection

st.title("Document Chunking, Topic Modeling, and Question Answering")

uploaded_file = st.file_uploader("Choose a JSON file", type="json")
if uploaded_file is not None:
    crawled_data = json.load(uploaded_file)

    # Step 1: Chunk data based on semantic similarity
    chunked_data = chunk_data(crawled_data)
    
    # Step 2: Topic modeling to organize chunks by topic
    topic_chunks = topic_modeling(chunked_data)
    
    # Encode topic chunks
    encoded_data = encode_topic_chunks(topic_chunks)
    
    # Create a Milvus collection
    collection = create_collection()
    
    # Insert vectors into MILVUS and create index
    insert_vectors(collection, encoded_data)
    create_index(collection)

    st.write("Data has been processed and indexed.")

    query = st.text_input("Enter your query:")
    if query:
        results = retrieve_and_rerank(query, topic_chunks, collection)
        for idx, (result, similarity_score) in enumerate(results, start=1):
            st.write(f"Rank {idx}: Similarity Score: {similarity_score:.4f}, Result ID: {result.id}")

        context = " ".join([chunk for chunk in list(dict(topic_chunks).values())[0]])
        question = st.text_input("Enter your question for the context:")
        if question:
            answer = answer_question(question, context)
            st.write(f"Question: {question}")
            st.write(f"Answer: {answer}")
