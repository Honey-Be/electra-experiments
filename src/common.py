import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

from electra_pytorch import Electra

from typing import Generic, TypeVar, OrderedDict, Union, Self, Tuple, Callable, Option

A = TypeVar('A', bound=nn.Module)
class LogitsAdapter(torch.nn.Module, Generic[A]):
        def __init__(self, adaptee: A):
            super().__init__()
            self.adaptee = adaptee

        def forward(self, *args, **kwargs):
            return self.adaptee(*args, **kwargs)[0]

Gen = TypeVar('Gen',bound=nn.Module)
Disc = TypeVar('Disc', bound=nn.Module)
class ElectraWrapper(Electra, Generic[Gen, Disc], metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def _tie_embeddings(cls, gen: Gen, disc: Disc) -> None:
        pass
    
    @classmethod
    @property
    @abstractmethod
    def backbone_type(cls) -> str:
        pass

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
        fix: Option[Callable[[Gen, Disc], None]] = None,
        **kwargs
    ):
        _generator_inner: Gen = Gen(config=model_generator)
        _discriminator_inner: Disc = Disc(config=model_generator)
        if fix:
            fix(_generator_inner, _discriminator_inner)
        else:
            Self._tie_embeddings(_generator_inner,_discriminator_inner)
        (_generator, _discriminator): Tuple[Union[LogitsAdapter, Gen], Union[LogitsAdapter, Disc]] = (LogitsAdapter(_generator_inner), LogitsAdapter(_discriminator_inner)) if wrap_to_logits_adapter else (Gen(model_generator), Disc(model_discriminator))
        super().__init__(
            _generator,
            _discriminator,
            num_tokens=len(vocab),
            mask_token_id=vocab[mask_token],
            pad_token_id=vocab[pad_token],
            mask_prob=mask_prob,
            mask_ignore_token_ids=list(
                map(
                    lambda tok: vocab[tok],
                    [class_token, separator_token]
                )
            ),
            random_token_prob=random_token_prob
        )
        self._generator = _generator
        self._discriminator = _discriminator
        self.distributed_enabled = distributed_enabled
                

    @abstractmethod
    def save_pretrained(self, output_dir: str) -> None:
        pass
    
    @property
    def generator(self) -> Union[LogitsAdapter[Gen], Gen]:
        return self._generator

    @property
    def discriminator(self) -> Union[LogitsAdapter[Disc], Disc]:
        return self._discriminator
    
    def try_to_distributed_model(self, rank, device: torch.device) -> Union[nn.parallel.DistributedDataParallel, Self]:
        if self.distributed_enabled:
            return nn.parallel.DistributedDataParallel(self.to(device), device_ids=[rank], find_unused_parameters=True)
        else:
            return self
