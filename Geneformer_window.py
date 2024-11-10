import math
import os, pickle
import scanpy as sc
import numpy as np
from collections import Counter
import logging
import warnings

import torch

import wandb
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from datasets import load_from_disk
from transformers import EarlyStoppingCallback
from transformers.training_args import TrainingArguments
# #############################################
# import torch.distributed as dist
# #############################################
from utils_geneformer import set_seed
from GenoHoption.Geneformer_window import graphTrainer
from GenoHoption.Geneformer_adaptor import SeqDataset, get_model, Geneformer_adaptor

os.environ["KMP_WARNINGS"] = "off"
# os.environ["WANDB_MODE"] = "offline"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
sys.path.insert(0, "../")
# from sc_foundation_evals.helpers.custom_logging import log
# log.setLevel(logging.INFO)

geneformer_data = "weights/Geneformer"
model_dir = os.path.join(geneformer_data, "default")
dict_dir = os.path.join(geneformer_data, "dicts")
import json
from types import SimpleNamespace
with open(os.path.join(model_dir, 'config.json'), 'r') as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

mode = 'Longformer'

for DATASET_NAME in ['ms']:
    if DATASET_NAME == 'ms':
        trainset_organ = load_from_disk("data/Real_data/ms/c_data.dataset")
        testset_organ = load_from_disk("data/Real_data/ms/filtered_ms_adata.dataset")
    elif DATASET_NAME == 'pancreas':
        trainset_organ = load_from_disk("data/Real_data/pancreas/demo_train.dataset")
        testset_organ = load_from_disk("data/Real_data/pancreas/demo_test.dataset")
    else:
        trainset_organ = load_from_disk("data/Real_data/myeloid/reference_adata.dataset")
        testset_organ = load_from_disk("data/Real_data/myeloid/query_adata.dataset")
    for SEED in [0,1,2,3,4]:
        for FRAC in [0.75]:
            hyperparameter_defaults = dict(
                seed=SEED,
                dataset_name=DATASET_NAME,
            )
            run = wandb.init(
                config=hyperparameter_defaults,
                project="Cell Clustering",
                name=f'Trail_{SEED}_{FRAC}_{DATASET_NAME}_32',
                group=f"{mode}_Geneformer",
                reinit=True,
                settings=wandb.Settings(start_method="fork"),
            )
            set_seed(SEED)

            def Disk2Dataset(trainset_organ, testset_organ):

                target_dict_list = []

                celltype_counter = Counter(trainset_organ["cell_type"])
                total_cells = sum(celltype_counter.values())
                cells_to_keep = [k for k, v in celltype_counter.items() if v > (0.00 * total_cells)]

                def if_not_rare_celltype(example):
                    return example["cell_type"] in cells_to_keep

                trainset_organ_subset = trainset_organ.filter(if_not_rare_celltype, num_proc=5)
                testset_organ_subset = testset_organ.filter(if_not_rare_celltype, num_proc=5)

                # shuffle datasets and rename columns
                trainset_organ_shuffled = trainset_organ_subset.shuffle(seed=0)
                testset_organ_shuffled = testset_organ_subset.shuffle(seed=0)
                trainset_organ_shuffled = trainset_organ_shuffled.rename_column("cell_type", "label")
                testset_organ_shuffled = testset_organ_shuffled.rename_column("cell_type", "label")

                # create dictionary of cell types : label ids
                target_names = list(Counter(trainset_organ_shuffled["label"]+testset_organ_shuffled["label"]).keys())
                target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))
                target_dict_list += [target_name_id_dict]

                # change labels to numerical ids
                def classes_to_ids(example):
                    example["label"] = target_name_id_dict[example["label"]]
                    return example

                labeled_trainset = trainset_organ_shuffled.map(classes_to_ids, num_proc=16)
                labeled_testset = testset_organ_shuffled.map(classes_to_ids, num_proc=16)

                ######################## Change different fold #########################
                random.seed(SEED)
                indices = list(range(len(labeled_trainset)))  # List of all sample indices
                num_train = round(len(labeled_trainset) * FRAC)  # Number of samples in the training set

                # Random sampling
                train_indices = random.sample(indices, num_train)
                eval_indices = list(set(indices) - set(train_indices))

                # Split the data set based on the results of random sampling
                labeled_train_split = labeled_trainset.select(train_indices)
                labeled_eval_split = labeled_trainset.select(eval_indices)

                # filter dataset for cell types in corresponding training set
                trained_labels = list(Counter(labeled_train_split["label"]).keys())
                ######################## Change different fold #########################

                def if_trained_label(example):
                    return example["label"] in trained_labels

                labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=16)
                labeled_test_split_subset = labeled_testset.filter(if_trained_label, num_proc=16)

                dataset_list = labeled_train_split
                evalset_list = labeled_eval_split_subset
                testset_list = labeled_test_split_subset

                return dataset_list, evalset_list, testset_list, len(target_name_id_dict.keys()), train_indices, eval_indices

            dataset_list, evalset_list, testset_list, config.num_labels, train_indices, eval_indices \
                = Disk2Dataset(trainset_organ, testset_organ)

            # set model parameters
            # max input size
            max_input_size = 2**11 # 2048

            # set training hyperparameters
            # max learning rate
            max_lr = 5e-5
            # how many pretrained layers to freeze
            freeze_layers = 0
            # batch size for training and eval
            geneformer_batch_size = 32
            logging_steps = round(len(trainset_organ)/geneformer_batch_size/0.01)
            # learning schedule
            lr_schedule_fn = "linear"
            # warmup steps
            warmup_steps = 500
            # number of epochs
            epochs = 10
            # optimizer
            optimizer = "adamw"

            # construct dataset & dataloader
            dataset_train = SeqDataset(dataset_list, dataset_name=DATASET_NAME, dataset_type='Train_Eval', indices=train_indices)

            dataset_eval = SeqDataset(evalset_list, dataset_name=DATASET_NAME, dataset_type='Train_Eval', indices=eval_indices)

            dataset_test = SeqDataset(testset_list, dataset_name=DATASET_NAME, dataset_type='Test')

            # initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model(model_dir, config, mode=mode)
            model = Geneformer_adaptor(config=config, original_model=model)
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                model = torch.nn.DataParallel(model)
            model = model.to(device)

            # define compute_metrics
            def compute_metrics(pred):
                print('enter')
                labels = pred.label_ids
                preds = pred.predictions.argmax(-1)
                # calculate accuracy and macro f1 using sklearn's function
                acc = accuracy_score(labels, preds)
                macro_f1 = f1_score(labels, preds, average='macro')
                precision = precision_score(labels, preds, average="macro")
                recall = recall_score(labels, preds, average="macro")
                return {
                  'accuracy': acc,
                  'macro_f1': macro_f1,
                  'recall': recall,
                  'precision': precision,
                }

            # set training arguments
            training_args = {
                "learning_rate": max_lr,
                "do_train": True,
                "do_eval": True,
                "evaluation_strategy": "epoch",
                "save_strategy": "epoch",
                "logging_steps": logging_steps,
                "disable_tqdm": False,
                "lr_scheduler_type": lr_schedule_fn,
                "warmup_steps": warmup_steps,
                "weight_decay": 0.001,
                "per_device_train_batch_size": geneformer_batch_size,
                "per_device_eval_batch_size": geneformer_batch_size,
                "num_train_epochs": epochs,
                "load_best_model_at_end": True,
                "output_dir":"save/output/",
                "metric_for_best_model":"eval_loss",
                'greater_is_better':False,
                "report_to":"wandb",
            }

            training_args_init = TrainingArguments(**training_args)

            class DictDataset(Dataset):
                def __init__(self, data_dict):
                    self.data_dict = data_dict
                    self.keys = list(data_dict.keys())
                    self.length = len(data_dict[self.keys[0]])

                def __len__(self):
                    return self.length

                def __getitem__(self, index):
                    return {key: self.data_dict[key][index] for key in self.keys}


            def dataset_attention_mask(dataset_train):
                item = dataset_train.data

                result = {}
                result['input_ids'] = item['input_ids']
                result['label'] = item['label']
                # result['attention_mask'] = []
                # for batch_data in dataset_train:
                #     a = batch_data['gene_graph']
                #     a.remove_nodes(0)
                #     result['attention_mask'].append(a.adjacency_matrix().to_dense().unsqueeze(dim=0)) # adjacency_matrix as attention mask
                dataset_train = DictDataset(result)
                return dataset_train

            dataset_train = dataset_attention_mask(dataset_train)
            dataset_eval = dataset_attention_mask(dataset_eval)
            dataset_test = dataset_attention_mask(dataset_test)

            trainer = graphTrainer(
                model=model,
                DATASET_NAME=DATASET_NAME,
                mode=mode,
                args=training_args_init,
                train_dataset=dataset_train,
                eval_dataset=dataset_eval,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
            )

            # train the cell type classifier
            trainer.train()
            predictions = trainer.predict(test_dataset=dataset_eval)
            wandb.log({'test/accuracy':predictions.metrics['test_accuracy'],
                       'test/macro_f1':predictions.metrics['test_macro_f1'],
                       'test/recall': predictions.metrics['test_recall'],
                       'test/precision': predictions.metrics['test_precision'],
                       })
# if __name__ == "__main__":
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '5678'
#     dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=2)
#     demo_basic()