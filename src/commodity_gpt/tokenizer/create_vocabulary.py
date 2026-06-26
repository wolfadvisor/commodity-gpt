import torch

def CreateVocabulary(text: str):
    """Builds a token vocabulary from input text and encodes it into a PyTorch tensor.

    Args:
        text (str): A raw text string containing words separated by spaces.

    Returns:
        tuple: A tuple containing:
            - tokens (torch.Tensor): A 1D tensor of word indices (dtype=torch.long).
            - vocabulary (dict): The generated word-to-index mapping dictionary.
    """
    # 1. Split text into individual words
    words = text.split()
    
     # 2. Build forward vocabulary automatically (word_to_id)
    word_to_id = {}
    for word in words:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)
            
    # 3. Professional Improvement: Build reverse vocabulary automatically (id_to_word)
    id_to_word = {idx: word for word, idx in word_to_id.items()}
            
    # 4. Encode text to integer index sequence
    encoded = [word_to_id[word] for word in words]
    
    # 5. Convert into PyTorch tensor
    tokens = torch.tensor(encoded, dtype=torch.long)
    
    # --- CHALLENGE OUTPUTS ---
    print("\n--- Challenge Results ---")
    print(f"Vocabulary Size  : {len(word_to_id)}")
    print(f"Forward Mapping  : {word_to_id}")
    print(f"Reverse Mapping  : {id_to_word}")
    print(f"Encoded sequence : {encoded}")
    print(f"Tensor Shape     : {tokens.shape}")
    print(f"PyTorch Tensor   : {tokens}\n")
    
    return tokens, word_to_id, id_to_word

    
    
