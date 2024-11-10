import os
from collections import Counter
import logging
import warnings

import torch
#########################################################
# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
########################################################################
import wandb
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import Trainer
from transformers import EarlyStoppingCallback
from transformers import BertForSequenceClassification
from transformers.training_args import TrainingArguments

from geneformer import DataCollatorForCellClassification
from utils import set_seed

os.environ["KMP_WARNINGS"] = "off"
os.environ["WANDB_MODE"] = "offline"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
sys.path.insert(0, "../")
from sc_foundation_evals.helpers.custom_logging import log
log.setLevel(logging.INFO)

geneformer_data = "weights/Geneformer"
model_dir = os.path.join(geneformer_data, "default")
dict_dir = os.path.join(geneformer_data, "dicts")

from datasets import load_from_disk

for FRAC in [0.75]:
    for DATASET_NAME in ['ms', 'pancreas', 'myeloid']:
        for SEED in [0,1,2,3,4]:
            if DATASET_NAME == 'ms':
                trainset_organ = load_from_disk("data/Real_data/ms/c_data.dataset")
                testset_organ = load_from_disk("data/Real_data/ms/filtered_ms_adata.dataset")
            elif DATASET_NAME == 'pancreas':
                trainset_organ = load_from_disk("data/Real_data/pancreas/demo_train.dataset")
                testset_organ = load_from_disk("data/Real_data/pancreas/demo_test.dataset")
            else:
                trainset_organ = load_from_disk("data/Real_data/myeloid/reference_adata.dataset")
                testset_organ = load_from_disk("data/Real_data/myeloid/query_adata.dataset")
            hyperparameter_defaults = dict(
                seed=SEED,
                dataset_name=DATASET_NAME,
            )
            run = wandb.init(
                config=hyperparameter_defaults,
                project="Cell Clustering",
                name=f'Trail_{SEED}_{DATASET_NAME}_32',
                group="Geneformer_new",
                reinit=True,
                settings=wandb.Settings(start_method="fork"),
            )
            set_seed(SEED)

            dataset_list = []
            evalset_list = []
            testset_list = []
            organ_list = []
            target_dict_list = []

            celltype_counter = Counter(trainset_organ["cell_type"])
            total_cells = sum(celltype_counter.values())
            cells_to_keep = [k for k, v in celltype_counter.items() if v > (0.00 * total_cells)]

            def if_not_rare_celltype(example):
                return example["cell_type"] in cells_to_keep

            trainset_organ_subset = trainset_organ.filter(if_not_rare_celltype, num_proc=5)
            testset_organ_subset = testset_organ.filter(if_not_rare_celltype, num_proc=5)

            # shuffle datasets and rename columns
            trainset_organ_shuffled = trainset_organ_subset.shuffle(seed=SEED)
            testset_organ_shuffled = testset_organ_subset.shuffle(seed=SEED)
            trainset_organ_shuffled = trainset_organ_shuffled.rename_column("cell_type", "label")
            testset_organ_shuffled = testset_organ_shuffled.rename_column("cell_type", "label")

            # create dictionary of cell types : label ids
            target_names = list(Counter(trainset_organ_shuffled["label"] + testset_organ_shuffled["label"]).keys())
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
            # labeled_test_split_subset = labeled_testset.filter(if_trained_label, num_proc=16)

            dataset_list += [labeled_train_split]
            evalset_list += [labeled_eval_split_subset]
            # testset_list += [labeled_test_split_subset]
            testset_list += [labeled_testset]

            # set model parameters
            # max input size
            max_input_size = 3000 # 2048

            # set training hyperparameters
            # max learning rate
            max_lr = 5e-5
            # how many pretrained layers to freeze
            freeze_layers = 0
            # # number gpus
            # num_gpus = 1
            # # number cpu cores
            # num_proc = 1
            # batch size for training and eval
            geneformer_batch_size = 32 # per device!!!!
            logging_steps = round(len(trainset_organ)/geneformer_batch_size/0.01)
            # learning schedule
            lr_schedule_fn = "linear"
            # warmup steps
            warmup_steps = 500
            # number of epochs
            epochs = 10
            # optimizer
            optimizer = "adamw"

            model = BertForSequenceClassification.from_pretrained(model_dir,num_labels=len(target_dict_list[0].keys()),output_attentions=False, output_hidden_states=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # DP
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                model = torch.nn.DataParallel(model)

            # map model
            model = model.to(device)

            def compute_metrics(pred):
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
                'gradient_checkpointing':True,
            }

            training_args_init = TrainingArguments(**training_args)

            cesu = True
            if cesu:
                import time
                from thop import profile
                from thop import clever_format

                total_flops = 0.
                total_params = 0.
                total_Time = 0.
                total_memory = 0.
                cesu_count = 10
                inputs = []
                for i, per in enumerate(dataset_list[0]):
                    inputs.append(torch.Tensor(per['input_ids']).unsqueeze(dim=0).long().to(device))
                    if i >= cesu_count-1:
                        break

                for j in range(cesu_count):
                    mem_before = torch.cuda.memory_allocated()
                    start_time = time.time()
                    flops, params = profile(model, (inputs[j],))
                    torch.cuda.synchronize()
                    end_time = time.time()
                    mem_after = torch.cuda.memory_allocated()
                    total_Time = total_Time + end_time - start_time

                    total_memory = (mem_after - mem_before) / (1024 ** 3) + total_memory

                    total_flops = flops + total_flops
                    total_params = params + total_params
                flops, params = clever_format([total_flops / cesu_count, total_params / cesu_count], "%.3f")
                print('FLOPs = ' + flops )
                print('Params = ' + params )
                print(f' GPU time used by method: {total_Time / 10} s')
                print(f'GPU memory used by method: {total_memory / 10} GB')
                print('Done')
                print('Done')
                break

            # create the trainer
            trainer = Trainer(
                model=model,
                args=training_args_init,
                data_collator=DataCollatorForCellClassification(),
                train_dataset=dataset_list[0],
                eval_dataset=evalset_list[0],
                compute_metrics=compute_metrics,
                # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
            )

            # train the cell type classifier
            trainer.train()
            predictions = trainer.predict(test_dataset=testset_list[0])
            wandb.log({'test/accuracy':predictions.metrics['test_accuracy'],
                       'test/macro_f1':predictions.metrics['test_macro_f1'],
                       'test/recall': predictions.metrics['test_recall'],
                       'test/precision': predictions.metrics['test_precision'],
                       })
