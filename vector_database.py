from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, IndexType
from chunking import model
# Connect to MILVUS server
connections.connect(host='localhost', port='19530')

def encode_topic_chunks(topic_chunks):
    encoded_chunks = []
    for topic, chunks in topic_chunks.items():
        chunk_embeddings = model.encode(chunks)
        for embedding in chunk_embeddings:
            encoded_chunks.append(embedding.tolist())
    return encoded_chunks

def create_collection():
    # Define the schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, "Collection for storing document embeddings")
    collection = Collection(name="my_collection", schema=schema)
    return collection

def create_index(collection, index_type=IndexType.IVF_FLAT, nlist=4096):
    index_params = {
      "metric_type": "L2",
      "index_type": index_type,
      "params": {"nlist": nlist}
    }
    collection.create_index(field_name="embedding", index_type=index_type, index_params=index_params)

def insert_vectors(collection, encoded_topic_chunks):
    # Insert vectors into MILVUS collection
    ids = collection.insert([encoded_topic_chunks])
    return ids
