import torch
import torch.nn as nn
import torch.nn.functional as F

from rich import print
from rich.traceback import install

from commodity_gpt.model.create_vocabulary import CreateVocabulary

install()

#Vocabulary 
word = """
    soybean corn coffe price increased decreased
    
"""
token, word_to_id, id_to_word = CreateVocabulary(word)
print(token, word_to_id, id_to_word)
vocab_size = len(word_to_id)


#last Token Representation

last_token = torch.tensor([
    0.25,1.10,-0.70
])

print("\nLast Token")
print(last_token)

#LM_HEAD

lm_head = nn.Linear(
    in_features=3,
    out_features= vocab_size
)

logits = lm_head(last_token)
print("\nLogits")
print(logits)

# PROBABILITIES
# ======================================

probabilities = F.softmax(
    logits,
    dim=-1
)

print("\nProbabilities")
print(probabilities)

# PREDICTION

predict_id = torch.argmax(
    probabilities
).item()

predicted_word = id_to_word[predict_id]

print("\nPrediction")
print(predicted_word)