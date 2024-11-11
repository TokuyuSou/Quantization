# This implementation is a minor modification of the original implementation in the Hugging Face evaluate library (https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py).
import datasets
import evaluate
import numpy as np
import torch
from evaluate import logging
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

_CITATION = """\
"""

_DESCRIPTION = """
Perplexity (PPL) is one of the most common metrics for evaluating language models.
It is defined as the exponentiated average negative log-likelihood of a sequence, calculated with exponent base `e`.
For more information, see https://huggingface.co/docs/transformers/perplexity
"""

_KWARGS_DESCRIPTION = """
Args:
    model (transformers.PreTrainedModel): Pretrained language model used for calculating Perplexity.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the model.
    predictions (list of str): input text, each separate text snippet
        is one list entry.
    batch_size (int): the batch size to run texts through the model. Defaults to 16.
    add_start_token (bool): whether to add the start token to the texts,
        so the perplexity can include the probability of the first word. Defaults to True.
    device (str): device to run on, defaults to 'cuda' when available
Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.
Examples:
    Example:
        >>> from datasets import load_dataset
        >>> perplexity = evaluate.load("perplexity", module_type="metric")
        >>> input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")['text'][:10]
        >>> input_texts = [s for s in input_texts if s != '']
        >>> results = perplexity.compute(
        ...                              model=my_model,
        ...                              tokenizer=my_tokenizer,
        ...                              predictions=input_texts)
        >>> print(round(results["mean_perplexity"], 2))
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Perplexity(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    def _compute(
        self,
        model,
        tokenizer,
        predictions,
        batch_size: int = 16,
        add_start_token: bool = True,
        device=None,
        max_length=None,
    ):
        if device is not None:
            assert device in [
                "gpu",
                "cpu",
                "cuda",
                "mps",
            ], "device should be either gpu, cpu, cuda, or mps."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = model.to(device)

        # If batch_size > 1, ensure tokenizer has a pad token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(
                tokenizer.special_tokens_map_extended.values()
            )
            assert (
                len(existing_special_tokens) > 0
            ), "Model must have at least one special token for padding."
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            assert (
                tokenizer.bos_token is not None
            ), "Model must have a BOS token if using add_start_token=True."
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        if add_start_token:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 1)
            ), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, inputs must be at least two tokens long."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * encoded_batch.size(0)
                ).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [
                        torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(
                            device
                        ),
                        attn_mask,
                    ],
                    dim=1,
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (
                    loss_fct(shift_logits.transpose(1, 2), shift_labels)
                    * shift_attention_mask_batch
                ).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def evaluate_perplexity(model, tokenizer, input_texts):
    perplexity_metric = Perplexity()
    results = perplexity_metric._compute(
        model=model,
        tokenizer=tokenizer,
        predictions=input_texts,
        batch_size=16,
        add_start_token=True,
        device=(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
    )
    return results
