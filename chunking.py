import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

nltk.download('punkt')

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

def chunk_data(data):
    chunked_data = []
    for url, content in data.items():
        paragraphs = content.get('paragraphs', [])
        if not paragraphs:
            continue

        # Tokenize paragraphs into sentences
        sentences = []
        for paragraph in paragraphs:
            sentences.extend(sent_tokenize(paragraph))

        # Encode sentences into embeddings
        sentence_embeddings = model.encode(sentences)

        # Use K-Means clustering to group sentences into chunks
        num_chunks = min(30, len(sentences))  # Limit to 30 chunks per document
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
