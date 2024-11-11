import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


# Function for text generation
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text: str,
    max_length: int = 50,
    num_return_sequences: int = 1,
) -> list[str]:
    """Generates text based on the input text using the specified model and tokenizer."""
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate text
    outputs = model.generate(
        inputs, max_length=max_length, num_return_sequences=num_return_sequences
    )

    # Decode outputs
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]

    return generated_texts
