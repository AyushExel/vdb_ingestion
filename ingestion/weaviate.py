import logging
import weaviate
import uuid
import weaviate.classes.config as wc
from tqdm import tqdm
import time
from .util import make_batch

def ingest_weaviate(batch_size, num_batches, vector_size):
    # Constants
    weaviate_collection_name = "weaviate_collection_00"

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Connect to Weaviate
    weaviate_client = weaviate.connect_to_embedded()

    # Create Weaviate collection
    weaviate_collection = weaviate_client.collections.create(
        name=weaviate_collection_name,
        properties=[
            wc.Property(name="item", data_type=wc.DataType.TEXT),
        ],
        vectorizer_config=None
    )

    try:
        weaviate_total_time = 0.0
        batch_times = []
        for _batch_index in tqdm(range(num_batches), desc="Processing batches"):
            ct = 0
            with weaviate_collection.batch.fixed_size(batch_size) as batch:

                batch_start_time = time.time()
                _batch = make_batch(batch_size, vector_size)
                vectors = _batch.column(0).to_pylist()
                items = _batch.column(1).to_pylist()
                for item, vector in zip(items, vectors):

                    batch.add_object(
                        properties={"item": item},
                        vector=vector
                    )

                    ct += 1

                    # If the number of vectors reached VECTORS_PER_BATCH threshold, it means the batch is injected with the desired number of vectors. (Ingestion of one batch is completed)
                    if ct % batch_size == 0:
                        batch_time = time.time() - batch_start_time
                        batch_times.append(batch_time)
                        weaviate_total_time += batch_time
                        logging.info(f"Batch {_batch_index + 1} inserted in {batch_time:.2f} seconds")

        weaviate_total_time = weaviate_total_time / 60
        print(f"\nTotal time to add {num_batches} batches of {batch_size} vectors of {vector_size} dimension: {weaviate_total_time:.2f} mins")

    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
        raise

    finally:
        weaviate_client.close()
