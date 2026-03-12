"""
Load patient embeddings into the Qdrant vector store.
"""
import json
import os

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "mimiciv_demo_emb"
OUTPUT_DIR = os.path.dirname(__file__) or "."

# Load embeddings and entity map
embeddings = np.load(os.path.join(OUTPUT_DIR, "patient_embeddings.npy"))
with open(os.path.join(OUTPUT_DIR, "entity_map.json")) as f:
    entity_map = json.load(f)

vector_size = embeddings.shape[1]

client = QdrantClient(url=QDRANT_URL)

# Create collection (recreate if it already exists)
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection '{COLLECTION_NAME}'")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)
print(f"Created collection '{COLLECTION_NAME}' (vector_size={vector_size}, distance=Cosine)")

# Build points
entities = sorted(entity_map.keys(), key=lambda k: entity_map[k])
points = []
for uri in entities:
    idx = entity_map[uri]
    patient_id = int(uri.split("/")[-1])
    points.append(PointStruct(
        id=idx,
        vector=embeddings[idx].tolist(),
        payload={"patient_id": patient_id, "uri": uri},
    ))

print(f"Upserting {len(points)} points...")
client.upsert(collection_name=COLLECTION_NAME, points=points)

info = client.get_collection(COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}': {info.points_count} points, vector size {info.config.params.vectors.size}")
print("Done!")
