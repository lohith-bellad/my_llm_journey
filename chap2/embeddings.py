from torch.utils.data import Dataset, DataLoader, dataloader
import tiktoken
import torch

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_len, "Number of tokenized inputs must be atleast equal to max_len+1"

        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i: i + max_len]
            target_chunk = token_ids[i+1: i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_len=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_len, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_len=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

vocab_size = 50257
output_dim = 256

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embeddings = embedding_layer(inputs)
print("token embeddings shape:",token_embeddings.shape)

pos_embeddings_layer = torch.nn.Embedding(4, output_dim)
pos_embeddings = pos_embeddings_layer(torch.arange(4))
print("position embeddings shape:",pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print("input embeddings shape:", input_embeddings.shape)
