import sys
import os
import math
import time
import argparse
from functools import partial
import json

import numpy as np

from tokenizers import Tokenizer
from rich.progress import Progress
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from icecream import ic

sys.path.append('.')

import layers as scr
from loss import cross_entropy_loss
from utils import compress_numpy_array, decompress_numpy_array

import warnings
warnings.filterwarnings('error')


ic.configureOutput(includeContext=True)
ic.disable()


class GoePT():

    def __init__(self,
                    vocab_size: int=8192,
                    context_length: int=256,
                    batch_size: int=64,
                    n_layer: int=6,
                    n_embd: int=384,
                    dropout: float=0.2,
                    lr: float=1e-3) -> None:

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.dropout = dropout
        self.lr = lr

        self.rng = np.random.default_rng()

        def weight_init(size):
            return self.rng.normal(size=size, loc=0.0, scale=0.02).astype(np.float32)

        def c_proj_weight_init(size):
            return self.rng.normal(size=size, loc=0.0, scale=0.02/math.sqrt(2 * self.n_layer)).astype(np.float32)

        def bias_init(size):
            return np.zeros(shape=size, dtype=np.float32)

        # Define lm_head first so we can pass its
        # weights_transposed property to the wte
        # embedding to implement weight tying

        self.lm_head = scr.Linear(self.n_embd,
                                    self.vocab_size,
                                    self.batch_size,
                                    bias=False,
                                    lr=self.lr,
                                    weight_init_func=weight_init,
                                    bias_init_func=bias_init)

        self.transformer = {
            "wte": scr.Embedding(self.vocab_size, self.n_embd, self.batch_size, self.lr, weight_external=self.lm_head.weight_transposed),
            "wpe": scr.Embedding(self.context_length, self.n_embd, self.batch_size, self.lr, init_func=weight_init),
            "drop": scr.Dropout(self.dropout),
            "h": [scr.Block(self.n_embd, self.context_length, 6, self.batch_size, self.lr, self.dropout, weight_init, c_proj_weight_init, bias_init) for _ in range(self.n_layer)],
            "ln_f": scr.LayerNorm(self.n_embd, weight_init_func=weight_init),
            }

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate

        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # assert id(self.transformer['wte'].weight) == id(self.lm_head.weight), "wte and lm_head must share the same weights in memory"



    def forward(self, idx, targets=None):
        b, t = idx.shape
        assert t <= self.context_length, f"Cannot forward sequence of length {t}, block size is only {self.context_length}"
        pos = np.arange(0, t, dtype=np.int64) # shape (t)

        # Forward the GPT model itself
        # Token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer['wte'].forward(idx)
        
        # Position embeddings of shape (t, n_embd)
        pos_emb = self.transformer['wpe'].forward(pos)

        # Main transformer
        x = self.transformer['drop'].forward(tok_emb + pos_emb)
        for block in self.transformer['h']:
            x = block.forward(x)
        x = self.transformer['ln_f'].forward(x)

        # Compute loss and return
        if targets is not None:
            # if we are given some desired targets also calculate the loss<
            logits = self.lm_head.forward(x)

            ic(logits.shape, targets.shape)
            logits_for_loss = logits.reshape(-1, logits.shape[-1])
            targets_for_loss = np.expand_dims(targets.reshape(-1), 1)
            targets_for_loss = scr.one_hot(targets_for_loss, 8192)

            loss = cross_entropy_loss(logits_for_loss, targets_for_loss)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head.forward(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


    def backward(self, x):
        raise NotImplementedError("Implement the nanoGPT backward path")


    def update(self):
        raise NotImplementedError("Implement the nanoGPT update")

    def state_dict(self):

        params_all = {'lm_head': [compress_numpy_array(self.lm_head.weight),
                                        compress_numpy_array(self.lm_head.bias)],
                        'wte': compress_numpy_array(self.transformer['wte'].weight),
                        'wpe': compress_numpy_array(self.transformer['wpe'].weight),
                        'ln_f': [compress_numpy_array(self.transformer['ln_f'].weight),
                                        compress_numpy_array(self.transformer['ln_f'].bias)]}

        for idx, block in enumerate(self.transformer['h']):
            params_all[f'block_{idx}'] = block.state_dict()

        state_dict = {
            'vocab_size': self.vocab_size,
            'context_length': self.context_length,
            'batch_size': self.batch_size,
            'n_layer': self.n_layer,
            'n_embd': self.n_embd,
            'dropout': self.dropout,
            'lr': self.lr,
            'params': params_all}

        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict):

        goe_pt = cls(state_dict['vocab_size'],
                            state_dict['context_length'],
                            state_dict['batch_size'],
                            state_dict['n_layer'],
                            state_dict['n_embd'],
                            state_dict['dropout'],
                            state_dict['lr'])

        goe_pt.lm_head.weight = decompress_numpy_array(state_dict['params']['lm_head'][0])
        goe_pt.lm_head.bias = decompress_numpy_array(state_dict['params']['lm_head'][1])

        goe_pt.transformer['wte'].weight = decompress_numpy_array(state_dict['params']['wte'])
        goe_pt.transformer['wpe'].weight = decompress_numpy_array(state_dict['params']['wpe'])

        goe_pt.transformer['ln_f'].weight = decompress_numpy_array(state_dict['params']['ln_f'][0])
        goe_pt.transformer['ln_f'].bias = decompress_numpy_array(state_dict['params']['ln_f'][1])

        for idx, block in enumerate(goe_pt.transformer['h']):
            block.load_params(state_dict['params'][f'block_{idx}'])

        return goe_pt



def read_datasets(split, data_dir, context_length, batch_size, rng):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = rng.integers(len(data) - context_length, size=(batch_size,))

    x = np.stack([(data[i:i+context_length].astype(np.int64)) for i in ix])
    y = np.stack([(data[i+1:i+1+context_length].astype(np.int64)) for i in ix])

    return x, y


def compute_gradient(target, prediction, one_hot_lookup):

    ic(prediction.shape)
    ic(target.shape)

    target = np.stack([one_hot_lookup[token] for token in target])

    ic(target.shape)

    return (prediction - target)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='NanoGPT from scratch')
    parser.add_argument('--data-dir', type=str,
                            default='datasets/tokenized/',
                            help='Dataset directory')
    parser.add_argument('--checkpoint-dir', type=str,
                                default='checkpoints/',
                                help='Checkpoint directory')
    parser.add_argument('--vocab-file', type=str,
                                default='models/tokenizers/goe_pt/goe_pt_tokenizer.json',
                                help='Vocabulary file')

    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--context-length', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=32, metavar='N')
    parser.add_argument('--eval-iters', type=int, default=200, metavar='N')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--eval-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.vocab_file)

    ic(tokenizer)

    np.random.seed(args.seed)

    model = GoePT(batch_size=args.batch_size, lr=args.lr)

    # state_dict = model.state_dict()
    # with open(os.path.join(args.checkpoint_dir, 'test_checkpoint.json'), mode='w', encoding='utf-8') as out_file:
    #     json.dump(state_dict, out_file)
    # with open(os.path.join(args.checkpoint_dir, 'test_checkpoint.json'), mode='r', encoding='utf-8') as in_file:
    #     state_dict = json.load(in_file)
    # model_loaded = GoePT.from_state_dict(state_dict)
    # ic(model_loaded)
    # exit()

    # training loop

    rng = np.random.default_rng()

    get_batch = partial(read_datasets,
                            data_dir=args.data_dir,
                            context_length=args.context_length,
                            batch_size=args.batch_size,
                            rng=rng)

    # Pre-generate one-hot vectors using the vocab size
    # for gradient computation
    one_hot_lookup = np.eye(8192)

    t0 = time.time()

    iter_num = 0

    best_val_loss = 1e9

    console = Console()
    status = console.status('[bold green]Starting training...', spinner='runner')
    progress_step = Progress(transient=True)

    with Live(Panel(Group(status, progress_step))):
        while True:
            progress_step.console.print(f"Starting epoch: {iter_num+1}")
            status.update(f'[bold green]Training epoch {iter_num+1} ...')

            X, Y = get_batch('train')

            task_id = progress_step.add_task('Training')

            for micro_step in progress_step.track(range(args.gradient_accumulation_steps),
                                                total=args.gradient_accumulation_steps,
                                                task_id=task_id):

                logits, loss = model.forward(X, Y)
                loss = loss/args.gradient_accumulation_steps # scale the loss to account for gradient accumulation

                # Get raw gradient
                raw_grad = compute_gradient(Y, logits, one_hot_lookup)

                # Continue backward
                grad = loss * raw_grad
                model.backward(grad)
                model.update()

                progress_step.console.print(f"Current local training loss: {loss:.5e}")
                progress_step.advance(task_id)

            progress_step.remove_task(task_id)

            # Evaluate the loss on train/val sets and write checkpoints

            if iter_num % args.eval_interval == 0:

                losses_dataset = {}

                for split in ['train', 'val']:
                    losses = np.zeros(args.eval_iters)

                    task_id = progress_step.add_task(f'{split.capitalize()} loss evaluation')

                    for k in progress_step.track(range(args.eval_iters),
                                                total=args.eval_iters,
                                                task_id=task_id):

                        X, Y = get_batch(split)

                        logits, loss = model.forward(X, Y)

                        losses[k] = loss.item()

                        progress_step.advance(task_id)

                    progress_step.remove_task(task_id)

                    losses_dataset[split] = losses.mean()

                if losses_dataset['val'] < best_val_loss:

                    status_update_string = f'Val loss decreased from {best_val_loss:.4f} to {losses_dataset["val"]:.4f}'

                    if iter_num > 0:
                        status_update_string += '. Saving checkpoint...'

                        status.update(status_update_string)

                        checkpoint_path = os.path.join(args.checkpoint_dir, f'goe_pt_iter_{iter_num}.json')

                        state_dict = model.state_dict()

                        with open(checkpoint_path, mode='w', encoding='utf-8') as out_file:
                            json.dump(state_dict, out_file)

                        status.update(f'Saved checkpoint under {checkpoint_path}')

                    else:
                        status.update(status_update_string)

                    best_val_loss = losses_dataset['val']

                status.update(f'[bold green]Training...\tstep {iter_num}: train loss {losses_dataset["train"]:.4f}\tval loss {losses_dataset["val"]:.4f}')

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if iter_num % args.log_interval == 0:
                lossf = loss.item()*args.gradient_accumulation_steps

                status.update(f'[bold green]Training...\tstep {iter_num}: loss {lossf:.4f}\ttime {dt*1000.:.2f} ms')

            iter_num += 1

            # termination conditions
            if iter_num > args.epochs:
                break


if __name__ == '__main__':
    main()
