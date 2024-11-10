import os
import json
import sys
import wandb
import pickle
from collections import Counter
sys.path.insert(0, "../")
import random
import time
import copy
import numpy as np
import warnings
from types import SimpleNamespace
os.environ["KMP_WARNINGS"] = "off"
os.environ["WANDB_MODE"] = "offline"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# dgls
from dgl.dataloading import GraphDataLoader as DataLoader
# My codes
from GenoHoption.Geneformer_adaptor import get_model, Geneformer_adaptor, SeqDataset, collate_fn
from datasets import load_from_disk
# import torchs
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

SEED = 0
FRAC = 0.75
############################# Load files ###############################################
geneformer_data = "weights/Geneformer"
model_dir = os.path.join(geneformer_data, "default")
dict_dir = os.path.join(geneformer_data, "dicts")
# Geneformer config
with open(os.path.join(model_dir, 'config.json'), 'r') as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
# Gene vocab
with open(os.path.join(dict_dir, "gene_name2id_dict.pkl"), 'rb') as f:
    vocab = pickle.load(f)

# fix seed
def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)
set_seed(SEED)
# batch size for training and eval
batch_size = 1
# max learning rate
max_lr = 5e-5
# how many pretrained layers to freeze
freeze_layers = 0
# warmup steps
warmup_steps = 500
# number of epochs
epochs = 10
# optimizer
optimizer = "adamw"
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print frequence
log_interval = 100
# max input size
max_input_size = 2 ** 11  # 2048

temperature = 5.
alpha=0.25
diffusion_mode = 1
iter_num = 1
FRAC = 0.75

############################# Load files ###############################################
for DATASET_NAME in ['ms', 'pancreas', 'myeloid']:
    for SEED in [0,1,2,3,4]:
        for FRAC in [0.75]:
            ############################# Construct Dataset ###############################################
            # old data
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
                ################# add diffusion_mode #####################
                diffusion_mode = diffusion_mode,
                ###########################################################
                ################# add hyper params #####################
                iter_num = iter_num,
                temperature = temperature,
                alpha = alpha,
                ###########################################################
            )
            config.alpha = alpha
            config.temperature = temperature
            config.iter_num = iter_num
            config.diffusion_mode = diffusion_mode
            run = wandb.init(
                config=hyperparameter_defaults,
                project=f"Cell Clustering",
                name=f'Trail_{SEED}_{FRAC}_{DATASET_NAME}_{batch_size}',
                group='Geneformer_w/o diffuser',
                reinit=True,
                settings=wandb.Settings(start_method="fork"),
            )
            wandb_config = wandb.config
            print(wandb_config)

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
                test_indices = list(range(len(labeled_testset)))

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

                if len(eval_indices) != labeled_eval_split_subset.shape:
                    print('Filted some data...')
                    new_eval_indices = []
                    for index, example in enumerate(labeled_eval_split):
                        if example["label"] in trained_labels:
                            new_eval_indices.append(eval_indices[index])
                    eval_indices = new_eval_indices

                # if len(test_indices) != labeled_test_split_subset.shape:
                #     print('Filted some data...')
                #     new_test_indices = []
                #     for index, example in enumerate(labeled_testset):
                #         if example["label"] in trained_labels:
                #             new_test_indices.append(test_indices[index])
                #     test_indices = new_test_indices

                dataset_list = labeled_train_split
                evalset_list = labeled_eval_split_subset
                # testset_list = labeled_test_split_subset
                testset_list = labeled_testset

                return dataset_list, evalset_list, testset_list, len(target_name_id_dict.keys()), train_indices, eval_indices, test_indices

            dataset_list, evalset_list, testset_list, config.num_labels, train_indices, eval_indices, test_indices = Disk2Dataset(trainset_organ, testset_organ)

            # construct dataset & dataloader
            dataset_type_train = 'Train_Eval'
            dataset_type_test = 'Test'
            dataset_train = SeqDataset(dataset_list, dataset_name=DATASET_NAME, dataset_type=dataset_type_train, indices=train_indices)
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=collate_fn)
            dataset_eval = SeqDataset(evalset_list, dataset_name=DATASET_NAME, dataset_type=dataset_type_train, indices=eval_indices)
            dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, collate_fn=collate_fn)
            dataset_test = SeqDataset(testset_list, dataset_name=DATASET_NAME, dataset_type=dataset_type_test, indices=test_indices)
            dataloader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate_fn)
            ############################# Construct Dataset ###############################################

            ############################# Import model ###############################################
            model = get_model(model_dir, config)
            model = Geneformer_adaptor(config=config, original_model=model)
            model = model.to(device)
            ############################# Import model ###############################################

            ############################# Train args ######################################################
            # Freeze the specified number of layers
            for i, param in enumerate(model.parameters()):
                if i < freeze_layers:
                    param.requires_grad = False

            #################### The same as Geneformer ###########################################
            total_steps = (len(dataset_train)// batch_size) * epochs if len(dataset_train) % batch_size == 0 else (len(dataset_train) // batch_size + 1) * epochs
            optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=0.001)
            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps,
                                                        )
            #################### The same as Geneformer ###########################################

            ############################# Train args ######################################################
            def train(model: nn.Module, loader: DataLoader) -> None:
                """
                Train the model for one epoch.
                """
                model.train()
                total_loss = 0.0
                start_time = time.time()
                num_batches = len(loader)
                for batch, batch_data in enumerate(loader):

                    optimizer.zero_grad()

                    bs = batch_data[0].shape[0]
                    attention_mask = batch_data[0].eq(vocab['<pad>'])
                    attention_mask = torch.cat([torch.tensor([[False]] * bs), attention_mask],dim=1)

                    input_ids = batch_data[0].long()
                    celltype_labels = batch_data[1].long()

                    input_g = batch_data[4]

                    input_ids = input_ids.to(device)
                    input_g = input_g.to(device)
                    attention_mask = attention_mask.to(device)
                    celltype_labels = celltype_labels.to(device)

                    logits = model(
                        input_ids=input_ids,
                        input_g=input_g,
                        attention_mask=attention_mask,
                    )
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, celltype_labels)
                    # labels:(bs,)
                    # logits:(bs, label_num)
                    wandb.log({"train/cls": loss.item()})
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()
                    if batch % log_interval == 0 and batch > 0:
                        lr = optimizer.param_groups[0]['lr']
                        ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                        cur_loss = total_loss / log_interval
                        print(
                            f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                            f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                            f"loss {cur_loss:5.2f} | "
                            )
                        total_loss = 0
                    start_time = time.time()


            def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
                """
                Evaluate the model on the evaluation data.
                """
                model.eval()
                total_loss = 0.0
                total_error = 0.0
                total_num = 0
                predictions = []
                with torch.no_grad():
                    for batch, batch_data in enumerate(loader):

                        bs = batch_data[0].shape[0]
                        attention_mask = batch_data[0].eq(vocab['<pad>'])
                        attention_mask = torch.cat([torch.tensor([[False]] * bs), attention_mask],dim=1)

                        input_ids = batch_data[0].long()
                        celltype_labels = batch_data[1].long()

                        input_g = batch_data[4]

                        input_ids = input_ids.to(device)
                        input_g = input_g.to(device)
                        attention_mask = attention_mask.to(device)
                        celltype_labels = celltype_labels.to(device)

                        output_values = model(
                            input_ids=input_ids,
                            input_g=input_g,
                            attention_mask=attention_mask,
                        )

                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(output_values, celltype_labels)

                        total_loss += loss.item() * batch_data[0].shape[1]
                        accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                        total_error += (1 - accuracy / batch_data[0].shape[1]) * batch_data[0].shape[1]
                        total_num += batch_data[0].shape[1]
                        preds = output_values.argmax(1).cpu().numpy()
                        predictions.append(preds)
                wandb.log(
                    {
                        "valid/cls": total_loss / total_num,
                        "valid/err": total_error / total_num,
                        "epoch": epoch,
                    },
                )
                if return_raw:
                    return np.concatenate(predictions, axis=0)
                return total_loss / total_num, total_error / total_num


            ## %% inference
            def test(model: nn.Module, loader: DataLoader) -> float:

                model.eval()

                celltypes_labels = np.array(loader.dataset.data['label'])

                predictions = evaluate(
                    model,
                    loader=loader,
                    return_raw=True,
                )

                # compute accuracy, precision, recall, f1
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                accuracy = accuracy_score(celltypes_labels, predictions)
                precision = precision_score(celltypes_labels, predictions, average="macro")
                recall = recall_score(celltypes_labels, predictions, average="macro")
                macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

                print(
                    f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
                    f"Macro F1: {macro_f1:.3f}"
                )

                results = {
                    "test/accuracy": accuracy,
                    "test/precision": precision,
                    "test/recall": recall,
                    "test/macro_f1": macro_f1,
                }

                return predictions, celltypes_labels, results


            best_val_loss = float("inf")
            best_avg_bio = 0.0
            best_model = None

            patient = 0
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                train(
                    model,
                    loader=dataloader_train,
                )
                val_loss, val_err = evaluate(
                    model,
                    loader=dataloader_eval,
                )
                elapsed = time.time() - epoch_start_time
                print("-" * 89)
                print(
                    f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                    f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
                )
                print("-" * 89)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model)
                    best_model_epoch = epoch
                    print(f"Best model with score {best_val_loss:5.4f}")
                    ##########patient mechanism############
                    patient = 0
                else:
                    patient += 1
                    ##########patient mechanism############

                ##########patient mechanism############
                if patient >= 1:
                    break
                ##########patient mechanism############

            print("best model epoch: ", best_model_epoch)

            ## Step 5: Inference with fine-tuned scGPT model
            predictions, labels, results = test(best_model, dataloader_test)
            wandb.log(results)