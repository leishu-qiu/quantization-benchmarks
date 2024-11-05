import numpy as np
import os
import json

def save_embeddings(embeddings, file_path):
    """
    Saves the embeddings to a file for future use.

    Parameters:
    embeddings (np.ndarray): The embeddings to save.
    file_path (str): The path to the file where embeddings will be saved.
    """
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    """
    Loads embeddings from a file.

    Parameters:
    file_path (str): The path to the file from which embeddings will be loaded.

    Returns:
    np.ndarray: The loaded embeddings.
    """
    return np.load(file_path)


def compute_recall_at_k(fp32_indices_list, binary_indices_list, k):
    if isinstance(fp32_indices_list, list):
        fp32_indices_list = np.array(fp32_indices_list)
    
    if isinstance(binary_indices_list, list):
        binary_indices_list = np.array(binary_indices_list)
    
    n_queries = fp32_indices_list.shape[0]
    total_recall = 0.0

    for i in range(n_queries):
        # Convert both sets of indices for the current query to sets for comparison
        fp32_set = set(fp32_indices_list[i][:k])  # Take the top k elements from fp32_indices_list
        binary_set = set(binary_indices_list[i][:k])  # Take the top k elements from binary_indices_list

        # Compute recall for the current query
        recall = len(fp32_set.intersection(binary_set)) / k
        total_recall += recall

    # Compute average recall rate
    average_recall_rate = total_recall / n_queries
    return average_recall_rate

# def compute_recall_at_k(fp32_indices_list, binary_indices_list):

#     # total_queries = len(fp32_indices_list) 
#     # total_overlap_rate = 0 

#     # for i in range(total_queries):
#     #     # fp32_top_100 = set(fp32_indices_list[i])  
#     #     # binary_top_k = set(binary_indices_list[i])

#     #     overlap = binary_top_k.intersection(fp32_top_100)
        
#     #     overlap_rate = len(overlap) / len(fp32_top_100)
#     #     total_overlap_rate += overlap_rate

#     # average_overlap_rate = total_overlap_rate / total_queries
#     # return average_overlap_rate
#     if isinstance(fp32_indices_list, list):
#         fp32_indices_list = np.array(fp32_indices_list)
#         n_queries, k = fp32_indices_list.shape
#     total_recall = 0.0
    
#     for i in range(n_queries):
#         # Convert both sets of indices for the current query to sets for comparison
#         fp32_set = set(fp32_indices_list[i])
#         binary_set = set(binary_indices_list[i])
        
#         # Compute recall for the current query
#         recall = len(fp32_set.intersection(binary_set)) / k
#         total_recall += recall
    
#     # Compute average recall rate
#     average_recall_rate = total_recall / n_queries
#     return average_recall_rate


def compute_recall(true_indices, pred_indices, k):
    correct = 0
    for true, pred in zip(true_indices, pred_indices):
        correct += len(set(true[:k]).intersection(set(pred[:k])))
    return correct / (len(true_indices) * k)
    

def save_metadata(dataset_size, METADATA_FILE):
    with open(METADATA_FILE, 'w') as f:
        json.dump({"dataset_size": dataset_size}, f)

# Function to load metadata
def load_metadata(METADATA_FILE):
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f).get("dataset_size", None)
    return None


    # Save the index to disk for future use
    # if isinstance(corpus_index, faiss.Index) and not os.path.exists(FAISS_INDEX_FILE):
    #     faiss.write_index(corpus_index, FAISS_INDEX_FILE)
    #     print(f"FAISS index saved to {FAISS_INDEX_FILE}")
    # This is a helper function to showcase how FAISS can be used with quantized embeddings.
    # You must either provide the `corpus_embeddings` or the `corpus_index` FAISS index, but not both.
    # In the first call we'll provide the `corpus_embeddings` and get the `corpus_index` back, which
    # we'll use in the next call. In practice, the index is stored in RAM or saved to disk, and not
    # recalculated for every query.

    # This function will 1) quantize the query embeddings to the same precision as the corpus embeddings,
    # 2) perform the semantic search using FAISS, 3) rescore the results using the full precision embeddings,
    # and 4) return the results and the search time (and perhaps the FAISS index).

    # `corpus_precision` must be the same as the precision used to quantize the corpus embeddings.
    # It is used to convert the query embeddings to the same precision as the corpus embeddings.
    # `top_k` determines how many results are returned for each query.
    # `rescore_multiplier` is a parameter for the rescoring step. Rather than searching for the top_k results,
    # we search for top_k * rescore_multiplier results and rescore the top_k results using the full precision embeddings.
    # So, higher values of rescore_multiplier will give better results, but will be slower.

    # `calibration_embeddings` is a set of embeddings used to calibrate the quantization of the query embeddings.
    # This is important only if you are using uint8 or int8 precision. In practice, this is used to calculate
    # the minimum and maximum values of each of the embedding dimensions, which are then used to determine the
    # quantization thresholds.

    # `rescore` determines whether to rescore the results using the full precision embeddings, if False & the
    # corpus is quantized, the results will be very poor. `exact` determines whether to use the exact search
    # or the approximate search method in FAISS.

    # 9. Output the results
    # print("Precision:", corpus_precision)
    # print(f"Search time: {search_time:.6f} seconds")
    # for query, result in zip(queries, results):
    #     print(f"Query: {query}")
    #     for entry in result:
    #         print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}")
    #     print("")



#     FAISS_INDEX_UBI_FILE = "faiss_index_ubinary.index"
# FAISS_INDEX_FP32_FILE = "faiss_index_fp32.index"


# if os.path.exists(FAISS_INDEX_FILE):
#     print(f"Loading FAISS index from {FAISS_INDEX_UBI_FILE}")
#     corpus_index = faiss.read_index(FAISS_INDEX_UBI_FILE)
#     print(f"Loading FAISS index from {FAISS_INDEX_UBI_FILE}")
#     corpus_index = faiss.read_index(FAISS_INDEX_UBI_FILE)
# else:
#     print("Creating FAISS index...")
#     corpus_index = faiss.IndexFlatL2(corpus_embeddings.shape[1])  # Use L2 distance for the index
#     corpus_index.add(corpus_embeddings)  # Add the corpus embeddings to the index
#     faiss.write_index(corpus_index, FAISS_INDEX_FILE)  # Save the index to disk
#     print(f"FAISS index saved to {FAISS_INDEX_FILE}")