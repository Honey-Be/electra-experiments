import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

from electra_pytorch import Electra as electra

from typing import Generic, TypeVar, OrderedDict, Union, Self, Tuple, Callable, Optional, Protocol, get_args

A = TypeVar('A', bound=nn.Module)
class LogitsAdapter(torch.nn.Module, Generic[A]):
        def __init__(self, adaptee: A):
            super().__init__()
            self.adaptee = adaptee

        def forward(self, *args, **kwargs):
            return self.adaptee(*args, **kwargs)[0]

Gen = TypeVar('Gen', bound=nn.Module)
Disc = TypeVar('Disc', bound=nn.Module)
class ElectraWrapper(electra, Generic[Gen, Disc], metaclass=ABCMeta):

    @abstractmethod
    def _tie_embeddings(self, gen: Gen, disc: Disc) -> None:
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
        gen_type: type[nn.Module] = Gen,
        disc_type: type[nn.Module] = Disc,
        fix: Optional[Callable[[Gen, Disc], None]] = None,
        **kwargs
    ):
        
        _generator_inner: Gen = gen_type(config=model_generator)
        _discriminator_inner: Disc = disc_type(config=model_generator)
        
        if fix is None:
            self._tie_embeddings(_generator_inner,_discriminator_inner)
        else:
            fix(_generator_inner, _discriminator_inner)
        _generator: Union[LogitsAdapter, Gen]
        _discriminator: Union[LogitsAdapter, Disc]
        _generator, _discriminator = (LogitsAdapter(_generator_inner), LogitsAdapter(_discriminator_inner)) if wrap_to_logits_adapter else (Gen(model_generator), Disc(model_discriminator))
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
        self._generator_inner: Gen = _generator_inner
        self._discriminator_inner: Disc = _discriminator_inner
                
    def forward(self, x,**kwargs):
        return electra.forward(self, x, **kwargs)
        

    @abstractmethod
    def save_pretrained(self, output_dir: str) -> None:
        pass
    
    @property
    def generator_inner(self) -> Gen:
        return self._generator_inner
    
    @property
    def discriminator_inner(self) -> Disc:
        return self._discriminator_inner
        
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
