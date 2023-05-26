import torch
import torch.nn as nn

from common import LogitsAdapter, ElectraWrapper

from transformers.models.fnet import FNetForMaskedLM, FNetForPreTraining
import torch.nn as nn
from typing import Tuple, OrderedDict, Optional, Self
import os

from typing import Callable

class FNetEmbeddingProjectionFixer(Callable[[FNetForMaskedLM, FNetForPreTraining],None]):
    def __init__(self, embedding_size: int, disc_config, gen_config=None, tie_embeddings: Optional[Callable[[FNetForMaskedLM, FNetForPreTraining], None]] = None):
        self._gen_config_given: bool = gen_config is not None
        self._embedding_size: int = embedding_size
        self._gen_hidden_size: int = gen_config.hidden_size
        self._disc_hidden_size: int = disc_config.hidden_size
        self._disc_vocab_size: int = disc_config.vocab_size
        self._disc_pad_token_id: int = disc_config.pad_token_id
        self._disc_max_position_embeddings: int = disc_config.max_position_embeddings
        self._disc_type_vocab_size: int = disc_config.type_vocab_size
        self._tie_embeddings: Optional[Callable[[FNetForMaskedLM, FNetForPreTraining], None]] = tie_embeddings
    
    def __call__(self, gen: FNetForMaskedLM, disc: FNetForPreTraining) -> None:
        disc.fnet.embeddings.word_embeddings = nn.Embedding(self._disc_vocab_size, self._embedding_size, padding_idx=self._disc_pad_token_id)
        disc.fnet.embeddings.position_embeddings = nn.Embedding(self._disc_max_position_embeddings, self._embedding_size)
        disc.fnet.embeddings.token_type_embeddings = nn.Embedding(self._disc_type_vocab_size, self._embedding_size)
        if self._gen_config_given:
            gen.fnet.embeddings.projection = nn.Linear(self.__embedding_size, self._gen_hidden_size)
        disc.fnet.embeddings.projection = nn.Linear(self.__embedding_size, self._disc_hidden_size)
        if self._gen_config_given and self._tie_embeddings:
            self._tie_embeddings(gen,disc)


class ElectraBackbonedWithFNet(ElectraWrapper[FNetForMaskedLM, FNetForPreTraining]):
    @classmethod
    def _tie_embeddings(cls, gen: FNetForMaskedLM, disc: FNetForPreTraining) -> None:
        (
            gen.fnet.embeddings.word_embeddings,
            gen.fnet.embeddings.position_embeddings,
            gen.fnet.embeddings.token_type_embeddings
        ) = (
            disc.fnet.embeddings.word_embeddings,
            disc.fnet.embeddings.position_embeddings,
            disc.fnet.embeddings.token_type_embeddings
        )
    
    @classmethod
    @property
    def backbone_type(cls) -> str:
        return "fnet"

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
        embedding_size: int = kwargs['embedding_size']
        fix: Callable[[FNetForMaskedLM, FNetForPreTraining, Callable[[FNetForMaskedLM, FNetForPreTraining], None]], None] = FNetEmbeddingProjectionFixer(
            embedding_size,
            gen_config=model_generator,
            disc_config=model_discriminator,
            tie_embeddings=lambda g,d: Self._tie_embeddings(g,d)
        )
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
            fix=fix,
            **kwargs
        )
    
    def save_pretrained(self, output_dir: str) -> None:
        self.discriminator.fnet.save_pretrained(output_dir)