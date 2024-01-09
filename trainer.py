from model import Model
import torch
from torch import nn
from dataset import CharDataset
from torch.utils.data.dataloader import DataLoader


## training function
def train_loop(dataloader, model, loss_fn, optimizer, max_iters = None, val_data = None):
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        if max_iters is not None and batch > max_iters:
            break
        X, y = X.to("cuda"), y.to("cuda")
        pred = model(X)
        loss = loss_fn(torch.permute(pred, (0,-1,-2)), y)

        ## Train and test evaluation
        if (batch) % 1000 == 0:
            train_loss = eval_loss(model, dataloader, loss_fn)
            val_loss = test_loop(val_data, model, loss_fn)
            print(f"Step:{batch} train loss:{loss.to('cpu').item()}, val loss:{val_loss.to('cpu').item()}")        
                
        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()




## Test function
@torch.no_grad
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    losses = torch.ones(num_batches)
    test_loss , correct = 0,0
    for i,(X,y) in enumerate(dataloader):
        X,y = X.to("cuda"), y.to("cuda")
        pred = model(X)
        losses[i] = loss_fn(torch.permute(pred,(0,-1,-2)), y)
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



    
    for X,y in dataloader:
        X,y = X.to("cuda"), y.to("cuda")
        pred = model(X)
        test_loss += loss_fn(torch.permute(pred, (0,-1,-2)), y).to("cpu")
    test_loss /= num_batches
    print(f"####Test loss: {test_loss.item()}")
         



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
val_dataset = CharDataset(config, data[:split_idx], type='test', train_dataset=train_dataset)
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
    train_loop(train_data_loader, model, loss_fn, optimizer, max_iters = 5000, val_data = val_data_loader)
    #test_loop(val_data_loader, model, loss_fn)
print("Done!")
torch.save(model, 'saved_model')


