import pickle
from src.data.CustomDataSet import CustomDataSet

import torch
import esm

def extract_features():
    # Load ESM-1b model
    _, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    with open('../../data/processed/whole_test.pkl', 'rb') as file:
        orig_data = pickle.load(file)

    data = [orig_data[i].sequence for i in range(len(orig_data))]
    _, _, batch_tokens = batch_converter(data)

    for i in range(len(orig_data) - 1):
        orig_data[i].add_embedding(batch_tokens[0][i])

    with open('../../data/processed/data_with_embeddings.pkl', 'wb') as file:
        pickle.dump(orig_data, file)

    return None

extract_features()
