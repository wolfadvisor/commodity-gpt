import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rich import print, inspect
from rich.traceback import install

install()

vocab_size = 12
embedding_dim = 8

embedding = nn.Embedding( num_embeddings= vocab_size, embedding_dim= embedding_dim)

tokens = torch.tensor([
    0,1,2,3
])

result = embedding(tokens)

print(result)
print(result.shape)
print(result[0])

#similarity 

v1 = torch.tensor([
    1.0,2.0,3.0
])

v2 = torch.tensor([
    1.0,2.0,3.0
])

v3 = torch.tensor([
    -1.0,-2.0,-3.0
])

def similarity(vector1, vector2, dim=0):

    sim_val = F.cosine_similarity(
        vector1,
        vector2,
        dim=dim
    ).item()

    print(f"Cosine Similarity: {sim_val:.4f}")

    if sim_val > 0.90:
        print("Very similar vectors")
    elif sim_val > 0.70:
        print("Similar vectors")
    elif sim_val > 0.30:
        print("Moderately similar vectors")
    elif sim_val >= 0:
        print("Weak similarity")
    else:
        print("Opposite directions")

    return sim_val


#Creating Embeddings

embedding = torch.tensor([
    [1., 0., 2.],  # soybean
    [1., 1., 2.],  # FOB
    [2., 1., 1.],  # price
    [0., 2., 1.]   # increased
])

print(embedding)
print(embedding.shape)
#Compute similarity Matrix

scores = embedding @ embedding.T
print(scores)

attention_weights = F.softmax(scores, dim=-1)
print(attention_weights)
print(attention_weights.sum(dim=-1))