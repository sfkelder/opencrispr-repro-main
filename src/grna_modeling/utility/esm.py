import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union

class ESM2:
    def __init__(self, model: str, device="cuda"):
        self.device = torch.device(device)

        # load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model).to(self.device)

        # set to eval (freeze)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def embed(self, sequence: Union[str, List[str]]):
        """Return per-residue embeddings for one or multiple sequences."""
        # tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.device)

        # run ESM2
        outputs = self.model(**inputs)

        # get hidden states
        # (batch, length, dim)
        embeddings = outputs.last_hidden_state

        # strip special BOS/EOS tokens if present
        if hasattr(self.tokenizer, "bos_token_id"):
            embeddings = embeddings[:, 1:-1, :]

        # if single string, squeeze batch dim
        if isinstance(sequence, str):
            embeddings = embeddings.squeeze(0)

        return embeddings
