import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import random
import logging
from time import time
from dataclasses import dataclass

import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from electra_pytorch import Electra

from openwebtext import arg
from openwebtext.dataset import load_owt, new_tokenizer, wrap_example_builder

logger = logging.getLogger(__name__)

from common import ElectraWrapper

from typing import TypeVar, Generic
from electra_default import *
from electra_backbone_fnet import *
from electra_backbone_mobilebert import *



########################################################################################################
## args

@dataclass
class OpenWebTextArgs:
    backbone_model_type: arg.Str = 'default'
    data_dir: arg.Str = 'data/openwebtext_features/v1'
    data_vocab_file: arg.Str = 'data/vocab_openwebtext_v1.txt'
    data_n_tensors_per_file: arg.Int = 2048
    data_max_seq_length: arg.Int = 128

    gpu: arg.Int = 0
    gpu_enabled: arg.Bool = True
    gpu_deterministic: arg.Bool = False
    gpu_mixed_precision: arg.Bool = False
    distributed_port: arg.Int = 8888
    distributed_enabled: arg.Bool = True
    distributed_world_size: arg.Int = 2

    model_mask_prob: arg.Float = 0.15

    opt_lr: arg.Float = 5e-4
    opt_batch_size: arg.Int = 128 // (distributed_world_size if distributed_enabled else 1)
    opt_warmup_steps: arg.Int = 10_000
    opt_num_training_steps: arg.Int = 60_000

    step_log: arg.Int = 10
    step_ckpt: arg.Int = 10_000
    
    model_generator: arg.Str = 'pretraining/openwebtext/generator.config.d/'
    model_discriminator: arg.Str = 'pretraining/openwebtext/discriminator.config.d/'

########################################################################################################
## train




def train(rank, args, variant: type[ElectraWrapper] = ElectraDefault):

    #######################
    ## distributed

    if args.distributed_enabled:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.distributed_world_size,
            rank=rank)
    if args.gpu_enabled:
        device = torch.device('cuda:{}'.format(rank))
    else:
        device = torch.device('cpu')

    is_master = True if not args.distributed_enabled else args.distributed_enabled and rank == 0


    print(f"passed: device = {device}")

    #######################
    ## preamble

    set_gpus(rank)
    set_seed(rank)
    set_cuda(deterministic=args.gpu_deterministic)

    output_dir = f'{args.output_dir}/{rank}'
    os.makedirs(output_dir, exist_ok=False)

    setup_logging(filename=f'{output_dir}/output.log', console=is_master)

    print(f"passed: preamble")

    #######################
    ## dataset

    tokenizer = new_tokenizer(vocab_file=args.data_vocab_file)
    vocab_size = len(tokenizer.vocab)
    ds_train = wrap_example_builder(dataset=load_owt(owt_dir=args.data_dir, n_tensors_per_file=args.data_n_tensors_per_file), vocab=tokenizer.vocab, max_length=args.data_max_seq_length)

    pad_token_id = tokenizer.vocab['[PAD]']
    mask_token_id = tokenizer.vocab['[MASK]']
    cls_token_id = tokenizer.vocab['[CLS]']
    sep_token_id = tokenizer.vocab['[SEP]']

    assert pad_token_id == 0
    assert cls_token_id == 101
    assert sep_token_id == 102
    assert mask_token_id == 103

    def collate_batch(examples):
        input_ids = torch.nn.utils.rnn.pad_sequence([example['input_ids'] for example in examples], batch_first=True, padding_value=pad_token_id)
        input_mask = torch.nn.utils.rnn.pad_sequence([example['input_mask'] for example in examples], batch_first=True, padding_value=pad_token_id)
        segment_ids = torch.nn.utils.rnn.pad_sequence([example['segment_ids'] for example in examples], batch_first=True, padding_value=pad_token_id)
        return input_ids, input_mask, segment_ids

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    ds_train_loader = iter(cycle(DataLoader(ds_train, batch_size=args.opt_batch_size, collate_fn=collate_batch)))


    print(f"passed: dataset")

    #######################
    ## model

    from transformers import AutoConfig, ElectraForMaskedLM, ElectraForPreTraining
    model_generator = AutoConfig.from_pretrained(args.model_generator)
    model_discriminator = AutoConfig.from_pretrained(args.model_discriminator)
    _model = variant(
        model_generator=model_generator,
        model_discriminator=model_discriminator,
        vocab=tokenizer.vocab,
        mask_prob=args.model_mask_prob,
        random_token_prob=0.,
        wrap_to_logits_adapter=True,
        distributed_enabled=args.distributed_enabled,
        embedding_size=128,
    )
    model = _model.try_to_distributed_model(rank=rank,device=device)
    discriminator = _model.discriminator_inner
    print(f"passed: model")

    #######################
    ## optimizer

    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step):
            learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
            learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
            return learning_rate
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def get_params_without_weight_decay_ln(named_params, weight_decay):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        return optimizer_grouped_parameters

    optimizer = torch.optim.AdamW(get_params_without_weight_decay_ln(model.named_parameters(), weight_decay=0.1), lr=args.opt_lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.opt_warmup_steps, num_training_steps=args.opt_num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.gpu_mixed_precision)

    print(f"passed: optimizer")

    #######################
    ## train

    t, steps_s, eta_m = time(), 0., 0

    print(f"args.data_max_seq_length : {args.data_max_seq_length}")

    for step in range(args.opt_num_training_steps+1):
        input_ids, input_mask, segment_ids = next(ds_train_loader)

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        print(f"... input_ids.shape[1] : {input_ids.shape[1]}")
        
        assert input_ids.shape[1] <= args.data_max_seq_length

        optimizer.zero_grad()

        print("... optimizer.zero_grad()")

        with torch.cuda.amp.autocast(enabled=args.gpu_mixed_precision):
            loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        print(f"... model in step {step} succeeded")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        metrics = {
            'step': (step, '{:8d}'),
            'loss': (loss.item(), '{:8.5f}'),
            'loss_mlm': (loss_mlm.item(), '{:8.5f}'),
            'loss_disc': (loss_disc.item(), '{:8.5f}'),
            'acc_gen': (acc_gen.item(), '{:5.3f}'),
            'acc_disc': (acc_disc.item(), '{:5.3f}'),
            'lr': (scheduler.get_last_lr()[0], '{:8.7f}'),
            'steps': (steps_s, '{:4.1f}/s'),
            'eta': (eta_m, '{:4d}m'),
        }

        if step % args.step_log == 0:
            sep = ' ' * 2
            logger.info(sep.join([f'{k}: {v[1].format(v[0])}' for (k, v) in metrics.items()]))

        if step > 0 and step % 100 == 0:
            t2 = time()
            steps_s = 100. / (t2 - t)
            eta_m = int(((args.opt_num_training_steps - step) / steps_s) // 60)
            t = t2

        if step % 200 == 0:
            logger.info(np.array2string(disc_labels[0].cpu().numpy(), threshold=sys.maxsize, max_line_width=sys.maxsize))
            logger.info(np.array2string(disc_pred[0].cpu().numpy(), threshold=sys.maxsize, max_line_width=sys.maxsize))

        if step > 0 and step % args.step_ckpt == 0 and is_master:
            os.makedirs(f'{args.output_dir}/ckpt/{step}', exist_ok=True)
            discriminator.save_pretrained(f'{args.output_dir}/ckpt/{step}/')
        
        print(f"... step {step} succeeded\n... ")

########################################################################################################
## preamble

def set_gpus(gpu):
    torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id, backbone_type: str):
    import datetime
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(f'output/', f'{backbone_type}/{exp_id}', t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(filename, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    import shutil
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


########################################################################################################
## main
SUPPORTED_BACKBONES: set[str] = {'default', 'mobilebert', 'fnet'}

def main():
    
    
   

    # args
    args = arg.parse_to(OpenWebTextArgs)

    args.backbone_model_type = args.backbone_model_type.lower()
    if args.backbone_model_type not in SUPPORTED_BACKBONES:
        print(f"Not supported backbone models: {args.backbone_model_type}",file=sys.stderr)
        return 1
    
    args.model_generator = args.model_generator + f'{args.backbone_model_type}.json'
    args.model_discriminator = args.model_discriminator + f'{args.backbone_model_type}.json'
    

    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id, args.backbone_model_type)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/ckpt', exist_ok=False)
    copy_source(__file__, output_dir)

    args.output_dir = output_dir
    args.exp_id = exp_id

    # distributed
    if args.distributed_enabled:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.distributed_port)
        torch.multiprocessing.spawn(train, nprocs=args.distributed_world_size, args=(args,))
    else:
        try:
            exitcode: int = 0
            if args.backbone_model_type == "fnet":
                train(rank=args.gpu, args=args, variant=ElectraBackbonedWithFNet)
            elif args.backbone_model_type == "mobilebert":
                train(rank=args.gpu, args=args, variant=ElectraBackbonedWithMobileBert)
            else:
                train(rank=args.gpu, args=args)
        except RuntimeError as e:
            print("Pretraining failed.\n", file=sys.stderr)
            print(f"{e}\n", file=sys.stderr)
            exitcode = 1
        else:
            exitcode = 0
        finally:
            return exitcode


if __name__ == '__main__':
    exit_value = main()
    if exit_value == 0:
        print("Pretraining succeeded.\n")
    sys.exit(exit_value)
