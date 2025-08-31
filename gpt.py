
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
n_emb = 32
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

@torch.no_grad()        #no backward gradient is calculated and hence memory efficient
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

class Head(nn.Module):
    #single head attention
    def __init__(self, head_size ):
        super().__init__()
        self.key = nn.Linear(n_emb , head_size)
        self.query = nn.Linear(n_emb, head_size)
        self.value = nn.Linear(n_emb, head_size)
        self.register_buffer("tril" , torch.tril(torch.ones(context_len , context_len)))        #tril is not a parameter hence we are registering it as a buffer


    def forward(self,x):
        b , t , c = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        #weight
        wei =q @ k.transpose(-2,-1) * c**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0 , float("-inf"))   #we are adding [:t, :t] becuase we want in case where t < context_length=8(fixed)
        wei = F.softmax(wei, dim = -1)
        out = wei @ v
        return out 

class MultiHeadAttention(nn.Module):
    #running in parallel
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_emb, n_emb)     #doubt --> why do we need to project again but we are already projecting in the feedforward network
    
    def forward(self , x):
        
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        out = self.proj(out)
        return  out  #concat over the last dimension so now we have more versatile features
                                                                    # what would happen if we concat over the second dimension
class FeedForwardNetwork(nn.Module):
    """ to let the differernt heads talk to each other to accumulate the features of the other and add the non_linearity component"""
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, n_emb), 
            nn.ReLU(),
            nn.Linear(n_emb, n_emb),
            )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_emb , num_head):
        super().__init__()
        head_size = n_emb//num_head     #why are we dividing 
        self.mha = MultiHeadAttention(num_head, head_size)
        self.ffwn = FeedForwardNetwork(n_emb)

    def forward(self, x):
        x = x + self.mha(x)         # adding residual connection for training deeper network
        x = x + self.ffwn(x)
        return x
        

#bigram_model
class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)        #n_emb is basically a projection into a dimension
        self.position_embedding_table = nn.Embedding(context_len, n_emb)        #we need to project it into n_emb space becuase without projecting it, learning the affinity between token in vocab_dim is quite intensive, so we make it learn in the small dimension
        # self.sa_head = MultiHeadAttention(num_head=4 , head_size= n_emb//4)     #because after concatenating we would be getting a 4 times the n_emb , hence dividing it by 4(doubt) so that final is 32 which we wanted
        # self.ffn = FeedForwardNetwork(n_emb)        #making proj of single/per token to proj of the same token, self attention is like communication , now they want to think upon that data,by calculating weight
        self.block = nn.Sequential(
            Block(n_emb, num_head=4 ),      #(doubt)--> after one block the model has better vector of projection of one token in embedding space
            Block(n_emb, num_head=4 ),
            Block(n_emb, num_head=4 ),
        )
        self.lm_head = nn.Linear(n_emb, vocab_size)


    def forward(self, idx, targets=None): 
        b , t = idx.shape
        tok_emb = self.token_embedding_table(idx)   #(batch , time , channel)
        pos_emb = self.position_embedding_table(torch.arange(t))        #numbers arranged
        x = tok_emb + pos_emb       #broadcasting
        # sa = self.sa_head(x)        #whenever we write something like self.sa_head(x) , it's forward method is called
        # ffn = self.ffn(sa)
        x = self.block(x)       #doubt(how sequntail is calculated in blocks)
        logits = self.lm_head(x)  #( batch , time , vocab_size)
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
            idx_cond = idx[:, -context_len:]        #because we can't have the input token more than context length, so we are basically cropping it, but if we are cropping it don't we loose some information , how do we deal with that? (doubt)
            logits, loss = self(idx_cond)   
            logits = logits[:,-1,:]         #because the last token of the batch has all the information regading the purana one
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

context_idx = torch.zeros((1,1), dtype = torch.long)
print(decode(model.generate(context_idx , max_new_tokens= 1000)[0].tolist()))
 

##more head ()

# Pros:
# - More diverse attention patterns
# - Better parallel processing
# - Can capture different types of relationships

# Cons:  
# - Each head has less representational capacity
# - Might miss complex patterns requiring more dimensions

## fewer head ()

# Pros:
# - Each head can learn richer representations
# - Better for complex pattern recognition

# Cons:
# - Less diversity in attention patterns
# - Fewer parallel computations
