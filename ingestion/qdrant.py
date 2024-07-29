from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import time
from .util import make_batch


def ingest_qdrant(batch_size, num_batches, vector_size):
    # Initialize the Qdrant client
    client = QdrantClient(":memory:")

    # Define your collection name
    collection_name = "my_collection"

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

    try:
        total_time = 0.0
        batch_times = []
        
        # Iterate over each batch with a progress bar
        for batch_index in tqdm(range(num_batches), desc="Processing batches"):

            batch = make_batch(batch_size, vector_size, batch_index * batch_size)
            # Extract vectors and items from the current batch
            vectors = batch.column(0).to_pylist()
            items = batch.column(1).to_pylist()

            batch_points = []
            for item, vector in zip(items, vectors):
                batch_points.append(
                    models.PointStruct(
                        id=int(item),
                        vector=vector,
                        payload={}  # You can add metadata here if needed
                    )
                )
            start_time = time.time()
            client.upsert(
                collection_name=collection_name,
                points=batch_points
            )
            end_time = time.time()

            batch_time = end_time - start_time
            batch_times.append(batch_time)
            total_time += batch_time
        
        total_time = total_time / 60
        print(f"Total time to add {num_batches} batches of {batch_size} vector of {vector_size} dimension: {total_time} mins")
        count = client.count(
        collection_name=collection_name,
        exact=True,
        )
        print(f"total rows added: {count}")

    except Exception as e:
        print(e)