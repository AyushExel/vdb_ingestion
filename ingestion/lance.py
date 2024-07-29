import time
import lancedb
from tqdm import tqdm
import bench.ingestion.lance as lance
from lancedb.pydantic import LanceModel, Vector
from bench.ingestion.util import make_batch
import pyarrow as pa


def ingest_lancedb(batch_size, num_batches, vector_size):
    db = lancedb.connect("./.lancedb")

    # Define the Item schema
    class Item(LanceModel):
        vector: Vector(vector_size)  
        item: str          

    # Create a table with the specified schema
    tbl = db.create_table("lancedb_table", schema=Item.to_arrow_schema(), mode="Overwrite")

    try:
        total_time = 0.0
        batch_times = []
        
        # Iterate over each batch with a progress bar
        for batch_index in tqdm(range(num_batches), desc="Processing batches"):
            # Generate a batch of random vectors and items
            batch = make_batch(batch_size, vector_size)
            table = pa.Table.from_batches([batch])
            start_time = time.time()
            tbl.add(table)
            end_time = time.time()

            batch_time = end_time - start_time
            batch_times.append(batch_time)
            total_time += batch_time
        
        total_time = total_time / 60
        print(f"Total time to add {num_batches} batches of {batch_size} vector of {vector_size} dimension: {total_time} mins")
        print(f"total rows added: {len(tbl)}")
    except Exception as e:
        print(e)