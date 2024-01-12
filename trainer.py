from model import Model
import torch
from torch import nn
from dataset import CharDataset
from torch.utils.data.dataloader import DataLoader
import argparse
from tqdm import tqdm


## training function
def train_loop(dataloader, model, loss_fn, optimizer, max_iters = None, val_data = None):
    model.train()
    total_steps = max_iters if max_iters is not None else len(dataloader)
    train_bar = tqdm(total=total_steps, desc='Train Step', position=0)
    for batch, (X,y) in enumerate(dataloader):
        if max_iters is not None and batch > max_iters:
            break        
        X, y = X.to("cuda"), y.to("cuda")
        pred = model(X)
        loss = loss_fn(torch.permute(pred, (0,-1,-2)), y)
        #print(f"step:{batch}, loss:{loss.item()}")

        ## Train and test evaluation
        if (batch) % 1000 == 0:
            #print("Entered into evaluation block")
            train_loss = eval_loss(model, dataloader, loss_fn)
            #print("Train evaluation done! going to test evaluation")
            val_loss = test_loop(val_data, model, loss_fn)
            train_bar.write(f"Step:{batch} train loss:{loss.to('cpu').item()}, val loss:{val_loss.to('cpu').item()}")        
                
        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_bar.update(1)




## Test function
@torch.no_grad
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    losses = torch.ones(num_batches)
    test_loss , correct = 0,0
    test_bar = tqdm(total=len(dataloader), desc='val loss step', position=1, leave=False) 
    for i, (X,y) in enumerate(dataloader):        
        X,y = X.to("cuda"), y.to("cuda")
        pred = model(X)
        losses[i] = loss_fn(torch.permute(pred,(0,-1,-2)), y)
        test_bar.update(1)
    model.train()
    return losses.mean()


@torch.no_grad
def eval_loss(model, dataloader, loss_fn, eval_iters=5):
    model.eval()
    losses = torch.ones(eval_iters)
    for i in range(eval_iters):
        X,y = next(iter(dataloader))
        X,y = X.to("cuda"), y.to("cuda")
        pred = model(X)
        loss = loss_fn(torch.permute(pred,(0,-1,-2)), y)
        losses[i] = loss
    model.train()
    return losses.mean().item()



#######  Main funciton ###############
if __name__ == '__main__':
    ### Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_save_path', help="enter the the path to save the model with .pt extension")
    parser.add_argument('max_iters', help="maximum iteration of a model")    
    args = parser.parse_args()
    model_save_path = args.model_save_path
    max_iters = int(args.max_iters)



    ## configuration settings
    class Config:
        def __init__(self):
            self.embedding_dim = 384
            self.n_heads = 6
            self.head_size = 512
            self.block_size = 256
            self.vocab_size = 65
            self.epochs = 20
            self.n_heads = 6
            self.n_blocks = 6

    config = Config()




    ### Prepare and load data

    # construct the entire dataset
    with open('data/input.txt', 'r') as f:
        data = f.read()

    # split dataset
    ratio = .9
    split_idx = int(len(data) * ratio)
    train_dataset = CharDataset(config, data[:split_idx], type='train')
    val_dataset = CharDataset(config, data[split_idx:], type='test', train_dataset=train_dataset)
    train_data_loader = DataLoader(
            train_dataset,
            #sampler=torch.utils.data.RandomSampler(val_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=True,
            pin_memory=True,
            batch_size=64,
            
        )
    val_data_loader = DataLoader(
            val_dataset,
            #sampler=torch.utils.data.RandomSampler(val_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=True,
            pin_memory=True,
            batch_size=64,
            
        )

    ## change required configuration settings
    config.vocab_size = train_dataset.get_vocab_size()
    config.head_size = config.embedding_dim // config.n_heads

    ## create the model
    model = Model(config.vocab_size, config.embedding_dim, config.block_size, config.head_size, config.n_heads, config.n_blocks)
    model = model.to("cuda")
    ## Trainign loop
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ## Train the data
    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_data_loader, model, loss_fn, optimizer, max_iters = max_iters, val_data = val_data_loader)
        #test_loop(val_data_loader, model, loss_fn)
    print("Done!")
    torch.save(model, model_save_path)