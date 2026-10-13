"""EnergyBERT adapter for the updated_OM matching pipeline."""

import os
import sys

_OM_UPDATED_DIR = os.path.dirname(os.path.abspath(__file__))
_OM_DIR = os.path.dirname(_OM_UPDATED_DIR)
_PROJECT_ROOT = os.path.dirname(_OM_DIR)
for _path in (_PROJECT_ROOT, _OM_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch
from wordembedding import bert_Energy_tsdae


class BertEnergy:
    """
    Wrapper around ``bert_Energy_tsdae`` exposing an ``embedding()`` API
    expected by ``utilizing.get_cosine_similarity``.
    """

    def __init__(self, model_name=None):
        """
        @param {str|None} model_name - Hugging Face id or local model path
        """
        self._model = bert_Energy_tsdae(model_name) if model_name else bert_Energy_tsdae()

    def embedding(self, sentences):
        """
        @param {list[str]} sentences - Texts to encode
        @returns {torch.Tensor} Mean-pooled embeddings, shape (n, hidden_size)
        """
        if not sentences:
            return torch.tensor([])
        return self._model._embedding(sentences)

    def sim(self, sentences_a, sentences_b):
        """
        @param {list[str]} sentences_a - Source terms
        @param {list[str]} sentences_b - Target terms
        @returns {torch.Tensor} Cosine similarity matrix
        """
        return self._model.sim(sentences_a, sentences_b)
