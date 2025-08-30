
import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r') as f:
    text = f.read()

#hyperparameter
torch.manual_seed(1337)
batch_size = 32
context_len= 8
iterations= 5000
lr= 1e-3
eval_iters= 200     #number of iterations for which evalution will occur on a sample from data
eval_interval = 100     #number of iterations after which evaluation will start

#enocder and decoder
chars = sorted(list(set(text)))
vocab_size = len(chars)
s_to_i = { ch:i for i,ch in enumerate(chars)} 
i_to_s = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [s_to_i[c] for c in s]
decode = lambda l: ''.join([i_to_s[i] for i in l])

#train and valid split
data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train = data[:n]
valid= data[n:]

#dataloader
def get_batch(split):
    data = train if split == "train" else valid
    i_x = torch.randint(len(data)-context_len,(batch_size,)) 
    x = torch.stack([data[i : i + context_len] for i in i_x])
    y = torch.stack([data[i+1:i+context_len+1] for i in i_x])
    return x,y

@torch.no_grad()        #no backward gradient is calculated
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train","valid"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits , loss = model(x,y)
            losses[k]= loss.item()
        out[split]= losses.mean()
    model.train()
    return out


#bigram_model
class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets=None): 
        
        logits = self.token_embedding_table(idx) 
        if targets is None:
            loss = None
        
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        
        return logits  , loss   #(batch_size, time_step , channel)
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)   
            logits = logits[:,-1,:] 
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1) #4 * 1
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr)

for iter in range(iterations):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step{iter}: train loss {losses ['train']:.4f}, val loss {losses ['valid']:.4f}")

    xb , yb = get_batch("train")

    #evaluate the loss
    logits , loss = model(xb , yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = idx= torch.zeros((1,1), dtype = torch.long)
print(decode(model.generate(context , max_new_tokens= 10)[0].tolist()[0]))