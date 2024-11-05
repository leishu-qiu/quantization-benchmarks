import time
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings, semantic_search_faiss
import faiss
import os
# import numpy as np
import json
import random
from utils import compute_recall_at_k, load_metadata, save_metadata

faiss.omp_set_num_threads(1)

INDEX_FOLDER = "faiss_indices"
os.makedirs(INDEX_FOLDER, exist_ok=True)

FAISS_INDEX_FILE_FP32 = os.path.join(INDEX_FOLDER, "faiss_index_fp32.index")
FAISS_INDEX_FILE_UBINARY = os.path.join(INDEX_FOLDER, "faiss_index_ubinary.index")
METADATA_FILE = os.path.join(INDEX_FOLDER, "metadata.json") #store dataset size


# 1. Load the quora corpus with questions
dataset = load_dataset("quora", split="train").map(
    lambda batch: {"text": [text for sample in batch["questions"] for text in sample["text"]]},
    batched=True,
    remove_columns=["questions", "is_duplicate"],
)
max_corpus_size = 100_00
saved_dataset_size = load_metadata(METADATA_FILE)
corpus = dataset["text"][:max_corpus_size]

number_of_queries = 100
queries = random.sample(corpus, min(number_of_queries, len(corpus)))
print(queries)
# queries = corpus[:10] 
# queries = [
#     "How do I become a good programmer?",
#     "How do I become a good data scientist?",
# ]


# 3. Load the model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")


# 4. Choose a target precision for the corpus embeddings
corpus_precision = "ubinary"


#6. Load or create FAISS index
if (os.path.exists(FAISS_INDEX_FILE_FP32) and os.path.exists(FAISS_INDEX_FILE_UBINARY)) and saved_dataset_size == max_corpus_size:
    print("Dataset Unchanged")
    # Load index
    print(f"Loading FAISS float32 index from {FAISS_INDEX_FILE_FP32}......")
    corpus_index_fp32 = faiss.read_index(FAISS_INDEX_FILE_FP32)

    print(f"Loading FAISS ubinary index from {FAISS_INDEX_FILE_UBINARY}......")
    corpus_index_ubi = faiss.read_index_binary(FAISS_INDEX_FILE_UBINARY)

    full_corpus_embeddings = None
    corpus_embeddings = None

else:
    # 5. Encode the corpus
    if saved_dataset_size is not None and saved_dataset_size != max_corpus_size:
        print("Size changed, recreating index...")
    full_corpus_embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=True, batch_size=16)
    corpus_embeddings = quantize_embeddings(full_corpus_embeddings, precision=corpus_precision)
    # NOTE: We can also pass "precision=..." to the encode method to quantize the embeddings directly,
    # but we want to keep the full precision embeddings to act as a calibration dataset for quantizing
    # the query embeddings. This is important only if you are using uint8 or int8 precision

    corpus_index_fp32 = None
    corpus_index_ubi = None

    save_metadata(max_corpus_size, METADATA_FILE)



# 7. Encode the queries using the full precision
start_time = time.time()
query_embeddings = model.encode(queries, normalize_embeddings=True)
print(f"Encoding time: {time.time() - start_time:.6f} seconds")


# corpus_index_fp32 = None

ground_truth_results, _, corpus_index_fp32 = semantic_search_faiss(
    query_embeddings,
    corpus_index=corpus_index_fp32,
    corpus_embeddings=full_corpus_embeddings if corpus_index_fp32 is None else None,
    corpus_precision="float32",
    top_k=100,
    rescore=False,
    output_index=True
)

# Save the float32 index after search
if not os.path.exists(FAISS_INDEX_FILE_FP32) and isinstance(corpus_index_fp32, faiss.Index):
    faiss.write_index(corpus_index_fp32, FAISS_INDEX_FILE_FP32)
    print(f"FAISS float32 index saved to {FAISS_INDEX_FILE_FP32}")



# 8. Perform semantic search using FAISS
for k in [1, 2, 3, 4, 5]:
    # corpus_index_ubi = None

    results, search_time, corpus_index_ubi = semantic_search_faiss(
        query_embeddings,
        corpus_index=corpus_index_ubi,
        corpus_embeddings=corpus_embeddings if corpus_index_ubi is None else None,
        corpus_precision=corpus_precision,
        top_k=100,
        calibration_embeddings=full_corpus_embeddings,
        rescore=corpus_precision != "float32",
        rescore_multiplier=k,
        output_index=True,
    )
    ground_truth_indices = [[entry['corpus_id'] for entry in res] for res in ground_truth_results]
    binary_indices = [[entry['corpus_id'] for entry in res] for res in results]  # Use `results` from binary search


    recall = compute_recall_at_k(ground_truth_indices, binary_indices, k=100)
    print(f"Recall for binary quantized embeddings at k = {k}: {recall:.4f}")



# Save the ubinary index after search
if not os.path.exists(FAISS_INDEX_FILE_UBINARY) and isinstance(corpus_index_ubi, faiss.IndexBinary):
    faiss.write_index_binary(corpus_index_ubi, FAISS_INDEX_FILE_UBINARY)
    print(f"FAISS ubinary index saved to {FAISS_INDEX_FILE_UBINARY}")


