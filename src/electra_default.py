import torch
import torch.nn as nn

from common import LogitsAdapter, ElectraWrapper

from transformers.models.electra import ElectraForMaskedLM, ElectraForPreTraining
import torch
import torch.nn as nn
from typing import Tuple, OrderedDict

class ElectraDefault(ElectraWrapper[ElectraForMaskedLM, ElectraForPreTraining]):
    def _tie_embeddings(self, gen: ElectraForMaskedLM, disc: ElectraForPreTraining) -> None:
        (
            gen.electra.embeddings.word_embeddings,
            gen.electra.embeddings.position_embeddings,
            gen.electra.embeddings.token_type_embeddings
        ) = (
            disc.electra.embeddings.word_embeddings,
            disc.electra.embeddings.position_embeddings,
            disc.electra.embeddings.token_type_embeddings
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
        gen_type = ElectraForMaskedLM,
        disc_type = ElectraForPreTraining,
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
            gen_type=gen_type,
            disc_type=disc_type,
            fix=None,
            **kwargs
        )

    def save_pretrained(self, output_dir: str) -> None:
        self.discriminator.electra.save_pretrained(output_dir)
