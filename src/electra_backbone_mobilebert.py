import torch
import torch.nn as nn

from common import LogitsAdapter, ElectraWrapper

from transformers.models.mobilebert import MobileBertForMaskedLM, MobileBertForPreTraining

import torch
import torch.nn as nn
from typing import Tuple, OrderedDict

class ElectraBackbonedWithMobileBert(ElectraWrapper[MobileBertForMaskedLM, MobileBertForPreTraining]):
    @classmethod
    def _tie_embeddings(cls, gen: MobileBertForMaskedLM, disc: MobileBertForPreTraining) -> None:
        (
            gen.mobilebert.embeddings.word_embeddings,
            gen.mobilebert.embeddings.position_embeddings,
            gen.mobilebert.embeddings.token_type_embeddings
        ) = (
            disc.mobilebert.embeddings.word_embeddings,
            disc.mobilebert.embeddings.position_embeddings,
            disc.mobilebert.embeddings.token_type_embeddings
        )

    def __init__(
        self,
        model_generator,
        model_discriminator,
        vocab: OrderedDict[str, int],
        mask_prob: float,
        random_token_prob: float=0.,
        wrap_to_logits_adapter: bool = False,
        distributed_enabled: bool = False,
        mask_token: str = '[MASK]',
        pad_token: str = '[PAD]',
        class_token: str = '[CLS]',
        separator_token: str = '[SEP]',
        **kwargs
    ):
        super().__init__(
            model_generator,
            model_discriminator,
            vocab,
            mask_prob,
            random_token_prob,
            wrap_to_logits_adapter,
            distributed_enabled,
            mask_token=mask_token,
            pad_token=pad_token,
            class_token=class_token,
            separator_token=separator_token,
            **kwargs
        )

    def save_pretrained(self, output_dir: str) -> None:
        self.discriminator.mobilebert.save_pretrained(output_dir)
