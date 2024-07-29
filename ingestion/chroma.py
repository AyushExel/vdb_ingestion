import chromadb
from tqdm import tqdm
import time
from .util import make_batch


def ingest_chroma(batch_size, num_batches, vector_size):
    # Initialize ChromaDB client
    client = chromadb.Client()

    # Create a collection in ChromaDB
    collection_name = "randomchromadb_collection"
    collection = client.create_collection(name=collection_name)
    try: # chroma can fail with larger batch sizes
        total_time = 0.0
        batch_times = []
        
        # Iterate over each batch with a progress bar
        for batch_index in tqdm(range(num_batches), desc="Processing batches"):
            batch = make_batch(batch_size, vector_size, batch_index * batch_size)
            # Extract vectors and items from the current batch
            vectors = batch.column(0).to_pylist()
            items = batch.column(1).to_pylist()

            start_time = time.time()
            collection.add(
                    embeddings=vectors,
                    ids=items
                )
            end_time = time.time()

            batch_time = end_time - start_time
            batch_times.append(batch_time)
            total_time += batch_time
        
        total_time = total_time / 60
        print(f"Total time to add {num_batches} batches of {batch_size} vector of {vector_size} dimension: {total_time} mins")
        print(f"total rows added: {collection.count()}")

    except Exception as e:
        print(e)
