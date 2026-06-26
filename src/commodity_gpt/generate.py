import torch

def Generate_Text(
        model,
        prompt,
        word_to_id,
        id_to_word,
        max_new_tokens,
        context_size=3
):
    "Generates text tokens one token at time"

    model.eval()

    words = prompt.split()

    tokens = [
        word_to_id[word]
        for word in words
    ]

    with torch.no_grad():

        for _ in range(max_new_tokens):
            #keep only the last context_size tokens

            context = tokens[-context_size:]

            #Pad if necessary

            while len(context) < context_size:
                context.insert(0,0)

            X = torch.tensor(
                [context],
                dtype=torch.long
            )

            logits = model(X)

            last_token_logits = logits[
                0,
                -1
            ]

            next_token_id = torch.argmax(
                last_token_logits
            ).item()

            tokens.append(
                next_token_id
            )

            generated_words = [
                id_to_word[token]
                for token in tokens
            ]

            return " ". join(
                generated_words
            )