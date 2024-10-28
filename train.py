import torch
import torch.nn as nn
from torch.distributed import init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, BackwardPrefetch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms 
from model import Model
from tokenizer import Tokenizer
from data.bookCorpus import Bookcorpus

# Initialize the process group for distributed training
def setup(rank, world_size):
    init_process_group("nccl", rank=rank, world_size=world_size)


# Training loop function
def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )

    # Initialize the model and wrap it with FSDP
    model = Model().to(rank)
    fsdp_model = FSDP(model, 
                      cpu_offload=CPUOffload(offload_params=True), 
                      backward_prefetch=BackwardPrefetch.BACKWARD_PRE)


    builder = Bookcorpus()
    builder.download_and_prepare()
    ds = builder.as_dataset(split='train')
    train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(ds, batch_size=32, sampler=train_sampler)



    t = Tokenizer()

    EPOCHS = 3
    LR = 1e-3

    opt = torch.optim.AdamW(fsdp_model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
        # Training loop
    fsdp_model.train()
    for epoch in range(5):  # loop over the dataset multiple times
        for batch in train_loader:
            opt.zero_grad()

            token_list = []
            max_len = 0
            for sentance in batch["text"]:
                cur_token_list = t.tokenize(sentance)
                if len(cur_token_list) > max_len:
                    max_len = len(cur_token_list)
                token_list.append(cur_token_list)

            for sentance_tokens in token_list:
                while len(sentance_tokens) < max_len:
                    sentance_tokens.append(t.end_of_file_index)
            token_list = torch.tensor(token_list)

            # Forward pass
            logits = fsdp_model(sentance_tokens)
            loss = loss_fn(logits[:, :max_len-1, :].transpose(-1, -2), token_list[:, 1:])

            # Backward pass
            loss.backward()
            opt.step()

        if __name__ == "__main__":
            print(f"Rank {rank}, Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}")

    # Clean up
    if __name__ == "__main__":
        torch.save(fsdp_model.module.state_dict(), "model.pt")
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    import os
    from torch.multiprocessing import spawn

    # Number of GPUs available
    world_size = torch.cuda.device_count()
    
    # Use torch.multiprocessing.spawn to launch processes on each GPU
    spawn(train, args=(world_size,), nprocs=world_size, join=True)