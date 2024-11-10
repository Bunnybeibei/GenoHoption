import dgl
import pickle
import os
import math
import torch
import logging
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union, Callable, List, Dict, Any
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator

######################### change vocab ################################
# Step1: Import
geneformer_data = "weights/Geneformer"
model_dir = os.path.join(geneformer_data, "default")
dict_dir = os.path.join(geneformer_data, "dicts")
token_name_id_path = os.path.join(dict_dir, "gene_name2id_dict.pkl")
with open(token_name_id_path, "rb") as f:
    vocab = pickle.load(f)
# Step2: Add cls
vocab['<cls>'] = len(vocab)
######################### change vocab ################################

class data_collator_attention_mask:
    def __init__(self):
        pass

    def __call__(self, examples):
        batch = {}
        temp = [torch.Tensor(i['input_ids']) for i in examples]
        batch['input_ids'] = pad_sequence(temp, batch_first=True, padding_value=vocab['<pad>'])

        batch['attention_mask'] = batch['input_ids'].eq(vocab['<pad>'])

        batch['labels'] = torch.stack([torch.Tensor([i['label']]) for i in examples]).squeeze(dim=-1)

        batch['input_ids'] = batch['input_ids'].long()
        batch['attention_mask'] = batch['attention_mask'].long()
        batch['labels'] = batch['labels'].long()

        return batch

class graphTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            DATASET_NAME: str=None,
            mode: str=None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(model,
                         args,
                         data_collator,
                         train_dataset,
                         eval_dataset,
                         tokenizer,
                         model_init,
                         compute_metrics,
                         callbacks,
                         optimizers,
                         preprocess_logits_for_metrics,
                         )
        self.mode = mode
        self.num_rand = 0
        if (self.mode == 'window') or (self.mode == 'bigbird'):
            if self.mode == 'bigbird':
                self.num_rand = 200
            if DATASET_NAME == 'ms':
                self.attention_window =[14 for _ in range(2048//14)]
            elif DATASET_NAME == 'pancreas':
                self.attention_window = [22 for _ in range(2048//22)]
            elif DATASET_NAME == 'myeloid':
                self.attention_window = [34 for _ in range(2048//34)]
            else:
                raise ValueError
        elif self.mode == 'Longformer':
            self.attention_window = [64 for _ in range(2048 // 64)]
        else:
            raise ValueError
        self.pad_token_id = vocab['<pad>']
        self._create_adj_mat()

    def _create_adj_mat(self):
        attention_window = (
            self.attention_window
            if isinstance(self.attention_window, int)
            else max(self.attention_window)
        )
        max_len = 2**11  # not the input sequence max len
        n_blocks = max_len // (attention_window // 2) - 1
        adj = np.zeros([max_len, max_len])

        # add local window att (overlap)
        for i in range(n_blocks):
            start = i * attention_window // 2
            end = start + attention_window
            if end > max_len:
                end = max_len
            adj[start:end, start:end] = 1

        # add random att
        num_random = max_len * self.num_rand

        idx = np.random.choice(range(max_len * max_len), num_random, replace=False)
        idx_x = idx % max_len
        idx_y = idx // max_len
        adj[idx_x, idx_y] = 1

        possible_seq_len = np.arange(attention_window, max_len + attention_window, attention_window)
        self.src_dst = {k: np.nonzero(adj[:k, :k]) for k in possible_seq_len}
        # 生成一个字典，可以直接提取指定长度的连接情况，一般是num_nodes * num_nodes + random_num

    def _pad_to_window_size(self, inputs):
        """
        它是对输入序列做右端padding,目的是让序列长度可以被attention window整除,这样在模型中计算self-attention时,可以对整个窗口进行并行计算。
        Args:
            inputs:

        Returns:

        """
        attention_window = (
            self.attention_window
            if isinstance(self.attention_window, int)
            else max(self.attention_window)
        )
        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = inputs["input_ids"].shape if inputs["input_ids"] is not None else inputs["attention_mask"].shape
        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logging.debug(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if inputs["input_ids"] is not None:
                inputs["input_ids"] = nn.functional.pad(inputs["input_ids"], (0, padding_len),
                                                        value=self.pad_token_id)
            inputs["attention_mask"] = nn.functional.pad(
                inputs["attention_mask"], (0, padding_len), value=False
            )  # no attention on the padding tokens
        return inputs

    def _from_adj_to_batched_graphs(self, input_ids):
        B = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        g_list = []
        for i in range(B):
            if self.mode == 'bigbird':
                src, dst = self.src_dst[seq_len]
                src = src + 1
                dst = dst + 1

                src_range = np.arange(1, seq_len + 1)
                src = np.concatenate([src, np.zeros(seq_len), src_range])

                dst_range = np.arange(1, seq_len + 1)
                dst = np.concatenate([dst_range, np.zeros(seq_len), dst])

                g = dgl.graph((src, dst))
            else:
                src, dst = self.src_dst[seq_len]
                g = dgl.graph((src, dst))
            g_list.append(g)
        batched_g = dgl.batch(g_list)
        return batched_g

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = self._pad_to_window_size(inputs)
        device = inputs["input_ids"].device
        batched_g = self._from_adj_to_batched_graphs(inputs["input_ids"]).to(device)
        inputs["input_g"] = batched_g
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = {}
        logits = model(**inputs)

        loss_fct = nn.CrossEntropyLoss()
        outputs['loss'] = loss_fct(logits, inputs["labels"])

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    @ torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        inputs = self._pad_to_window_size(inputs)
        device = inputs["input_ids"].device
        batched_g = self._from_adj_to_batched_graphs(inputs["input_ids"]).to(device)
        inputs["input_g"] = batched_g

        outputs = {}
        logits = model(**inputs)

        loss_fct = nn.CrossEntropyLoss()
        outputs['loss'] = loss_fct(logits, inputs["labels"])

        if prediction_loss_only:
            return (outputs['loss'], None, None)

        return (outputs['loss'], logits, inputs["labels"])


    def get_train_dataloader(self):
        num_workers = min(len(os.sched_getaffinity(0)), self.args.train_batch_size // 2)
        data_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=data_collator_attention_mask(),
        )
        return data_loader

    def get_eval_dataloader(self, eval_dataset: Dataset = None):
        num_workers = min(len(os.sched_getaffinity(0)), self.args.eval_batch_size // 2)
        data_loader = DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=data_collator_attention_mask(),
        )
        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        num_workers = min(len(os.sched_getaffinity(0)), self.args.eval_batch_size // 2)

        data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=data_collator_attention_mask()
        )
        return data_loader


class Trainer_attention_mask(Trainer):

    def get_train_dataloader(self):
        num_workers = min(len(os.sched_getaffinity(0)), self.args.train_batch_size // 2)
        data_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=data_collator_attention_mask(),
        )
        return data_loader

    def get_eval_dataloader(self, eval_dataset: Dataset = None):
        num_workers = min(len(os.sched_getaffinity(0)), self.args.eval_batch_size // 2)
        data_loader = DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=data_collator_attention_mask(),
        )
        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        num_workers = min(len(os.sched_getaffinity(0)), self.args.eval_batch_size // 2)

        data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=data_collator_attention_mask()
        )
        return data_loader