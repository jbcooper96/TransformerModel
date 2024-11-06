import torch
import os
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from model import Model
from tokenizer import Tokenizer
from data.bookCorpus import Bookcorpus
from torch.distributed.elastic.multiprocessing.errors import record

# Initialize the process group for distributed training
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("gloo", rank=rank, world_size=world_size)


# Training loop function
@record
def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Initialize the model and wrap it with FSDP
    weight_file = "model.pt"
    model = Model().to(rank)
    model.load_state_dict(torch.load(weight_file, weights_only=True))
    fsdp_model = DDP(model, device_ids=[rank])
     


    builder = Bookcorpus()
    builder.download_and_prepare()
    ds = builder.as_dataset(split='train[10%:20%]')
    train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(ds, batch_size=1000, sampler=train_sampler)



    t = Tokenizer()

    EPOCHS = 3
    LR = 1e-4

    opt = torch.optim.AdamW(fsdp_model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss().to(rank)
        # Training loop
    fsdp_model.train()
    for epoch in range(5):  # loop over the dataset multiple times
        for i, batch in enumerate(train_loader):
            opt.zero_grad()

            token_list = []
            sequence_length = 500
            for sentance in batch["text"]:
                try:
                    cur_token_list = t.tokenize(sentance)
                    
                    token_list += cur_token_list
                except:
                    print("Tokenizing Error")

            inputs = []
            while len(token_list) > 0:
                if len(token_list) > sequence_length:
                    inputs.append(token_list[:sequence_length])
                    token_list = token_list[sequence_length:]
                else:
                    token_list = []
            
            if rank == 0:
                print(len(inputs))

            inputs = torch.tensor(inputs)

            # Forward pass
            logits = fsdp_model(inputs.to(rank))
            loss = loss_fn(logits[:, :sequence_length-1, :].transpose(-1, -2).to(rank), inputs[:, 1:].to(rank))
            if rank == 0:
                print(loss.item())
            # Backward pass
            loss.backward()
            opt.step()
            if i % 200 == 0 and rank == 0:
                print("SAVING")
                torch.save(fsdp_model.module.state_dict(), "model.pt")

        

    # Clean up
    if rank == 0:
        torch.save(fsdp_model.module.state_dict(), "model.pt")
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    print(__name__)
    from torch.multiprocessing import spawn

    # Number of GPUs available
    world_size = torch.cuda.device_count()
    print(world_size)
    
    # Use torch.multiprocessing.spawn to launch processes on each GPU
    spawn(train, args=(world_size,), nprocs=world_size, join=True)