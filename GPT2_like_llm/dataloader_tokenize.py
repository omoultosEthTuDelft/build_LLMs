import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# print(torch.__version__)
# print(tiktoken.__version__)

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):   # sliding window to chunk the text into overlapping sequences of max_length
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids) # Returns the total number of rows in the dataset

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]  # Returns a single row from the dataset
    

def create_dataloader_v1(txt, 
                         batch_size=4, 
                         max_length=256,
                         stride=128, 
                         shuffle=True, 
                         drop_last=True,
                         num_workers=0
                         ):
    
    tokenizer = tiktoken.get_encoding('gpt2') # to initialize the tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) # to create the dataset
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  # if true, drops the last batch if it's shorter than the specified batch_size to prevent loss spikes in training
        num_workers=num_workers
    )
    return dataloader


if __name__ == '__main__':

    DOC_PATH = '/Users/omoultos/coding/build_LLMs/the-verdict.txt'
    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(
        raw_text, 
        batch_size=8, 
        max_length=4,
        stride=4, 
        shuffle=False   
    )

    data_iter = iter(dataloader) # to convert the dataloader into a Python iterator
    # first_batch = next(data_iter)
    # second_batch = next(data_iter)
    # print(first_batch)
    # print(second_batch)
    inputs, targets = next(data_iter)
    print(f'Inputs:\n{inputs}')
    print(f'\nTargets:\n{targets}')
