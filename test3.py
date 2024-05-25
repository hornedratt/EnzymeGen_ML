import pickle
from src.data.CustomDataSet import CustomDataSet

def test_esm1_msa1b_alphabet():
    import torch
    import esm

    # Load ESM-1b model
    _, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    batch_converter = alphabet.get_batch_converter()


    with open('data/processed/whole_test.pkl', 'rb') as file:
        test = pickle.load(file)

    print(len(test))

    data = [ test[0].sequence ]
    _, _, batch_tokens = batch_converter(data)
    # expected_tokens = torch.tensor(
    #     [
    #         [
    #             [0, 20, 15, 11, 7, 10, 16, 6],
    #             [0, 15, 5, 4, 11, 10, 5, 12],
    #             [0, 15, 5, 5, 12, 8, 16, 16],
    #         ]
    #     ]
    # )
    # assert torch.allclose(batch_tokens, expected_tokens)
    print(batch_tokens)
    print(len(batch_tokens[0, 0]), len(test[0].sequence[1]))

test_esm1_msa1b_alphabet()