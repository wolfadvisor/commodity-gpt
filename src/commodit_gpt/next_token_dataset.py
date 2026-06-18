import torch
from rich import print
from vocabulary import CreateVocabulary


text ="""
    soybean FOB price incresead
"""

token, word_to_id, id_to_word = CreateVocabulary(text)

print(f'Tokens:{token}')

X = []
y = []

for i in range (1, len(token)):
    X.append(token[:i])
    y.append(token[i])

print(f'\nInputs:\n')
print(X)

print(f'\nTargets:\n')
print(y)