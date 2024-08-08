import torch
import torch.nn.functional as F
words = open('names.txt', 'r').read().splitlines()

chars = sorted(set(''.join(words)))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos ={i:s for s,i in stoi.items()}
xs, ys = [], []
for word in words[:1]:
    chs = ['.'] + list(word) + ['.']
    for char1,char2 in zip(chs,chs[1:]):
        ix1 = stoi[char1]
        ix2 = stoi[char2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn(27,27,generator=g,requires_grad=True) #(27,27)

#forward pass
for k in range(100):
    xenc = F.one_hot(xs, num_classes=27).float() #(5,27)
    logits = xenc @ W #raw output layer
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True) # softMax
    loss = -probs[torch.arange(num),ys].log().mean() + 0.01 * (W**2).mean() # (W**2).mean() => regularization
    print(loss.item())
    #backward pass
    W.grad = None
    loss.backward()

    #gradient descent
    W.data += -10 * W.grad

