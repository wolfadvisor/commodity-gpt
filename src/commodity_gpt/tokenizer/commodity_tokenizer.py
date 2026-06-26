from commodity_gpt.tokenizer.tokenizer.create_vocabulary import create_vocabulary
from rich.traceback import install

install()

words = """
    soybean FOB price increased
    corn CIF price decreased
    sugar IC45 export stable
    coffee export increased
"""

new_vocab = create_vocabulary(text = words)

print(new_vocab)