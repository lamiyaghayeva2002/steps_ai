
## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repository.git
    cd your-repository
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure that you have a Milvus server running locally. You can start Milvus using Docker:
    ```bash
    docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:v2.0.0-rc8-20210924-0201
    ```

2. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

3. Upload a JSON file containing the document data via the Streamlit interface. The JSON file should be structured as follows:
    ```json
    {
        "url1": {
            "paragraphs": [
                "Paragraph 1 text.",
                "Paragraph 2 text.",
                ...
            ]
        },
        "url2": {
            "paragraphs": [
                "Paragraph 1 text.",
                "Paragraph 2 text.",
                ...
            ]
        },
        ...
    }
    ```

4. After uploading the file, the application will:
    - Chunk the document data into semantically similar sentences.
    - Perform topic modeling to group the chunks by topics.
    - Encode the topic chunks and store them in the Milvus vector database.
    - Create an index for efficient retrieval.

5. You can then enter a query to retrieve and re-rank relevant chunks based on the query.

6. Enter a specific question related to the retrieved chunks to get an answer using the QA model.

## Modules

### chunking.py
- **Functionality**: Chunks the document data into semantically similar sentences using Sentence Transformers and K-Means clustering.

### topic_modeling.py
- **Functionality**: Performs topic modeling on the chunked data using Latent Dirichlet Allocation (LDA).

### vector_database.py
- **Functionality**: Encodes topic chunks, creates a Milvus collection, and inserts vectors into the collection. Also creates an index for efficient search.

### retrieval.py
- **Functionality**: Expands the query using synonyms, retrieves initial candidates using BM25, and re-ranks the results based on similarity.

### question_answering.py
- **Functionality**: Uses a pre-trained question-answering model from Hugging Face to answer questions based on the provided context.

## Contributing

Feel free to open issues or submit pull requests if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
