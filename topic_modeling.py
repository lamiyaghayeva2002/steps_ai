from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
import numpy as np
from chunking import model  # Import the model from chunking.py

def topic_modeling(data):
    texts = []
    for item in data:
        texts.extend(item['chunks'])

    # Use LDA for topic modeling
    lda = LatentDirichletAllocation(n_components=10, random_state=0)
    doc_topic_dists = lda.fit_transform(abs(model.encode(texts)))

    topic_chunks = defaultdict(list)
    for i, topic_dist in enumerate(doc_topic_dists):
        topic_chunks[np.argmax(topic_dist)].append(texts[i])

    return topic_chunks
