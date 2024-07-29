import time
import pyarrow as pa
import numpy as np
import logging

def make_batch(vectors_per_batch, vector_size, item_start_id=0):
    vectors = [np.random.rand(vector_size).astype(np.float32).tolist() for _ in range(vectors_per_batch)]
    items = [str(vectors_per_batch + j + 1 + item_start_id ) for j in range(vectors_per_batch)]
    
    return pa.RecordBatch.from_arrays(
        [
            pa.array(vectors, pa.list_(pa.float32(), vector_size)),
            pa.array(items),
        ],
        ["vector", "item"],
    )