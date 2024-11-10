# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
from scipy.sparse import issparse
import copy
from pathlib import Path
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils_scBERT import *
from datetime import datetime
from time import time
import torch.multiprocessing as mp
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

import wandb
import warnings
os.environ["KMP_WARNINGS"] = "off"
os.environ["WANDB_MODE"] = "offline"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(DATASET_NAME, SEED):
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help='Master addr for dist finetune.')
    parser.add_argument("--master_port", type=str, default="29503", help='Master port for dist finetune.')
    parser.add_argument("--world_size", type=int, default=2, help='Number of GPUs.')
    parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    parser.add_argument("--gene_num", type=int, default=None, help='Number of genes.') # 16906, if not supplied, will take the number of genes in the supplied training data
    parser.add_argument("--epochs", type=int, default=10, help='Number of epochs.')
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("--batch_size", type=int, default=1, help='Number of batch size.')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
    parser.add_argument("--grad_acc", type=int, default=1, help='Number of gradient accumulation.')
    parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
    parser.add_argument("--pos_embed_g2v", default='True', action='store_true', help='Using Gene2vec encoding or not (default no unless this arg is supplied).')
    parser.add_argument("--g2v_file", type=str, default='weights/scBERT/gene2vec_16906.npy', help='File containing Gene2vec embeddings')
    parser.add_argument("--data_path", type=str, default='data/Real_data/c_data.h5ad', help='Path of data for finetune.')
    parser.add_argument("--model_path", type=str, default='weights/scBERT/panglao_pretrain.pth', help='Path of pretrained checkpoint to load.')
    parser.add_argument("--ft_ckpt", action="store_true", help="Add this flag if continuing to train an already finetuned model.")
    parser.add_argument("--ckpt_dir", type=str, default='save/scBERT/', help='Directory for saving checkpoints.')
    parser.add_argument("--nreps", type=int, default=3, help='Number of replicates for each data split experiment.')
    #parser.add_argument("--sampling_fracs", type=list, default=[1.0, 0.75, 0.5, 0.25, 0.1], help='List of fractions of training data to sample for sample efficiency experiments.') #passing a list doesn't actually work
    parser.add_argument("--debug", default=False, action="store_true", help="Debug setting: saves to new dir.")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%b-%d-%H:%M:%S")

    
    mp.spawn(
        distributed_finetune,
        args=(args, timestamp, DATASET_NAME, SEED),
        nprocs=args.world_size,
        join=True,
    )


def distributed_finetune(rank, args, timestamp, DATASET_NAME, SEED):

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION = args.grad_acc
    LEARNING_RATE = args.learning_rate
    VALIDATE_EVERY = args.valid_every
    CLASS = args.bin_num + 2
    POS_EMBED_USING = args.pos_embed_g2v
    PATIENCE = 10
    UNASSIGN_THRES = 0.0
    SAMPLING_FRACS = [0.75] #arg doesn't work currently

    ################ DDP ##########################
    is_master = rank == 0
    master_addr = args.master_addr
    master_port = args.master_port
    world_size = args.world_size
    ##########################################

    ### CLASSES FROM ORIGINAL CODE ###

    class SCDataset(Dataset):
        def __init__(self, data, label):
            super().__init__()
            self.data = data
            self.label = label

        def __getitem__(self, index):
            #rand_start = random.randint(0, self.data.shape[0]-1)
            full_seq = self.data[index].toarray()
            full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            full_seq = torch.from_numpy(full_seq).long() #long() converts to int64
            full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device) #this is the CLS token ?
            seq_label = self.label[index]
            return full_seq, seq_label

        def __len__(self):
            return self.data.shape[0]

    class Identity(torch.nn.Module):
        def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
            super(Identity, self).__init__()
            self.conv1 = nn.Conv2d(1, 1, (1, 200))
            self.act = nn.ReLU()
            self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
            self.act1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
            self.act2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

        def forward(self, x):
            x = x[:,None,:,:]
            x = self.conv1(x)
            x = self.act(x)
            x = x.view(x.shape[0],-1)
            x = self.fc1(x)
            x = self.act1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.act2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

    cur_time = time()
    setup_process(rank, master_addr, master_port, world_size)
    device = torch.device("cuda:{}".format(rank))

    print("Set up distributed processes...")

    dataset_name = DATASET_NAME
    if dataset_name == "ms":
        data_dir = Path("data/Real_data")  # RB
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype(
            "category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata.var["gene_name"], inplace=True)
        adata = adata.concatenate(adata_test, batch_key="str_batch")

    if dataset_name == "pancreas":  # RB
        data_dir = Path("data/Real_data/pancreas")
        adata = sc.read(data_dir / "demo_train.h5ad")
        adata_test = sc.read(data_dir / "demo_test.h5ad")
        adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata = adata.concatenate(adata_test, batch_key="str_batch")

    if dataset_name == "myeloid":
        data_dir = Path("data/Real_data/myeloid")
        adata = sc.read(data_dir / "reference_adata.h5ad")
        adata_test = sc.read(data_dir / "query_adata.h5ad")
        adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata = adata.concatenate(adata_test, batch_key="str_batch")

    if args.debug:
        debug_seq_len = 5000
        adata = adata[:1000,:debug_seq_len]
        GRADIENT_ACCUMULATION = 1

    label_dict, label = np.unique(np.array(adata.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    class_num = np.unique(label, return_counts=True)[1].tolist()
    #class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num]) #doesn't get used anywhere
    class_weight = torch.tensor([1/x for x in class_num])  #use this simpler weighting
    # label = torch.from_numpy(label)
    adata.obs["celltype_id"] = label

    if issparse(adata.X):
        adata.X = adata.X.toarray()

    data = adata.X[:,:1501]
    if args.gene_num is not None:
        SEQ_LEN = args.gene_num + 1
    else:
        SEQ_LEN = data.shape[1] + 1 # num_genes + 1

    adata_test = adata[adata.obs["str_batch"] == "1"]
    adata = adata[adata.obs["str_batch"] == "0"]

    data = adata.X[:,:1501]
    label = adata.obs["celltype_id"].tolist()
    label = np.array(label)
    label = torch.from_numpy(label)

    data_test = adata_test.X
    label_test = adata_test.obs["celltype_id"].tolist()
    label_test = np.array(label_test)
    label_test = torch.from_numpy(label_test)
    del adata, adata_test

    # create new model
    def instantiate_new_model():
        # create new model
        model = PerformerLM(
            num_tokens=CLASS,
            dim=200,
            depth=6,
            max_seq_len=SEQ_LEN,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=POS_EMBED_USING,
            g2v_file=args.g2v_file
        )
        model = model.to(device)

        # Load checkpoint onto correct rank
        checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
        consume_prefix_in_state_dict_if_present(checkpoint['model_state_dict'], "module.")
        if args.ft_ckpt:
            print("Loaded finetuned ckpt...")
            model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            cur_epoch = checkpoint['epoch']
            # Load optimizer
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Load scheduler
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("Loaded pretrained model...")
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0]).to(device)
            model = model.to(device)
            cur_epoch = 0

        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.norm.named_parameters():
            param.requires_grad = True
        for name, param in model.performer.net.layers[
            -1].named_parameters():  # make last layers of performer trainable during fine tuning
            param.requires_grad = True
        for name, param in model.to_out.named_parameters():
            param.requires_grad = True

        # optimizer
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=LEARNING_RATE,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9
        )

        return (model, optimizer, scheduler, cur_epoch)

    model, optimizer, scheduler, cur_epoch = instantiate_new_model()
    ####################### DDP ########################################
    model = DDP(model, device_ids=[device], output_device=device)
    ####################################################################

    try:
        # Control sources of randomness - for each run k, different seed is used
        # this effects parameter initialization & the subsampling of the [fixed] training set
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        for frac in SAMPLING_FRACS:
            hyperparameter_defaults = dict(
                seed=SEED,
                dataset_name=DATASET_NAME,
                fraction=frac,
            )
            if is_master:
                run = wandb.init(
                    config=hyperparameter_defaults,
                    project="Cell Clustering",
                    name=f'Trail_{SEED}_{frac}_{DATASET_NAME}',
                    group="scBERT",
                    reinit=True,
                    settings=wandb.Settings(start_method="fork"),
                )

            ####################### DDP ########################################
            dist.barrier()
            ####################################################################

            #implement class weights in loss to handle class imbalance
            loss_fn = nn.CrossEntropyLoss(weight=class_weight).to(device)

            ####################### DDP ########################################
            dist.barrier()
            ####################################################################
            trigger_times = 0
            max_acc = 0.0

            ###################### DDP ########################################
            # attempt to seed dataloader - this is required for true reproducibility
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            ###################################################################

            g = torch.Generator()
            g.manual_seed(SEED)

            #downsample training set
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED) #same val set across all runs
            for index_train, index_val in sss.split(data, label):
                index_train_small = np.random.choice(index_train, round(index_train.shape[0]*frac), replace=False) # different random subset will be chosen with each k
                data_train, label_train = data[index_train_small], label[index_train_small]
                train_dataset = SCDataset(data_train, label_train)
                data_val, label_val = data[index_val], label[index_val]
                val_dataset = SCDataset(data_val, label_val)
            test_dataset = SCDataset(data_test, label_test)
            #############################DDP###################################
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, worker_init_fn=seed_worker, generator=g)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, worker_init_fn=seed_worker, generator=g)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, worker_init_fn=seed_worker, generator=g)
            ###################################################################
            print("Loaded data...")

            for i in range(cur_epoch+1, EPOCHS+1):
                print("{} iterations in train dataloader per epoch".format(len(train_loader)))
                ###################### DDP ##############################
                train_loader.sampler.set_epoch(i)
                #########################################################
                model.train()
                dist.barrier()
                running_loss = 0.0
                cum_acc = 0.0
                for index, (data_t, labels_t) in tqdm(enumerate(train_loader)):
                    index += 1
                    data_t, labels_t = data_t.to(device), labels_t.to(device)
                    if index % GRADIENT_ACCUMULATION != 0:
                        with model.no_sync():
                            logits = model(data_t)
                            loss = loss_fn(logits, labels_t)
                            loss.backward()
                    if index % GRADIENT_ACCUMULATION == 0:
                        logits = model(data_t)
                        loss = loss_fn(logits, labels_t)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                        optimizer.step()
                        optimizer.zero_grad()
                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    final = softmax(logits)
                    final = final.argmax(dim=-1)
                    pred_num = labels_t.size(0)
                    correct_num = torch.eq(final, labels_t).sum(dim=-1)
                    cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
                    # break
                epoch_loss = running_loss / index
                epoch_acc = 100 * cum_acc / index
                epoch_loss = get_reduced(epoch_loss, device, 0, world_size) # dest_device set to rank
                epoch_acc = get_reduced(epoch_acc, device, 0, world_size)
                if is_master:
                    print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
                dist.barrier()
                scheduler.step()


                if i % VALIDATE_EVERY == 0:
                    model.eval()
                    dist.barrier()
                    running_loss = 0.0
                    predictions = []
                    truths = []
                    with torch.no_grad():
                        for index, (data_v, labels_v) in enumerate(val_loader):
                            index += 1
                            data_v, labels_v = data_v.to(device), labels_v.to(device)
                            logits = model(data_v)
                            loss = loss_fn(logits, labels_v)
                            running_loss += loss.item()
                            softmax = nn.Softmax(dim=-1)
                            final_prob = softmax(logits)
                            final = final_prob.argmax(dim=-1)
                            final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                            predictions.append(final)
                            truths.append(labels_v)
                            # break
                        del data_v, labels_v, logits, final_prob, final
                        # gather
                        predictions = distributed_concat(torch.cat(predictions, dim=0),
                                                         len(val_sampler.dataset),
                                                         world_size)
                        truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset),
                                                    world_size)
                        no_drop = predictions != -1
                        predictions = np.array((predictions[no_drop]).cpu())
                        truths = np.array((truths[no_drop]).cpu())
                        cur_acc = accuracy_score(truths, predictions)
                        f1 = f1_score(truths, predictions, average='macro')
                        val_loss = running_loss / index
                        val_loss = get_reduced(val_loss, device, 0, world_size)
                        if is_master:
                            print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f} | Accuracy: {cur_acc:.3f} ==')
                            print(confusion_matrix(truths, predictions))
                            print(classification_report(truths, predictions, labels=np.arange(len(label_dict)), target_names=label_dict.tolist(), digits=4))

                            duration = time() - cur_time
                            cur_time = time()

                            wandb.log({'train/cls':epoch_loss})
                            wandb.log({'train/Accuracy':epoch_acc})
                            wandb.log({'valid/cls':val_loss})
                            wandb.log({'valid/Accuracy':cur_acc})
                            wandb.log({'valid/F1':f1})

                        if cur_acc > max_acc:
                            max_acc = cur_acc
                            trigger_times = 0
                            best_model = copy.deepcopy(model)
                            best_model_epoch = i
                            print(f"Best model with score {max_acc:5.4f}")
                        else:
                            trigger_times += 1
                            if trigger_times > PATIENCE:
                                break
                del predictions, truths
                break

            # Test ###########################################################
            model.eval()
            dist.barrier()
            running_loss = 0.0
            predictions = []
            truths = []
            with torch.no_grad():
                for index, (data_v, labels_v) in enumerate(test_loader):
                    index += 1
                    data_v, labels_v = data_v.to(device), labels_v.to(device)
                    logits = model(data_v)
                    loss = loss_fn(logits, labels_v)
                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    final_prob = softmax(logits)
                    final = final_prob.argmax(dim=-1)
                    final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                    predictions.append(final)
                    truths.append(labels_v)
                    # break
                del data_v, labels_v, logits, final_prob, final
                # gather
                predictions = distributed_concat(torch.cat(predictions, dim=0),
                                                 len(test_sampler.dataset),
                                                 world_size)
                truths = distributed_concat(torch.cat(truths, dim=0), len(test_sampler.dataset),
                                            world_size)
                no_drop = predictions != -1
                predictions = np.array((predictions[no_drop]).cpu())
                truths = np.array((truths[no_drop]).cpu())
                cur_acc = accuracy_score(truths, predictions)
                f1 = f1_score(truths, predictions, average='macro')
                print('Done')
                results = {
                    "test/accuracy": cur_acc,
                    "test/macro_f1": f1,
                }
                if is_master:
                    wandb.log(results)
            # Test ###########################################################
    except Exception as e:
        print(e)
        pass #so that cleanup() occurs with or without error
    cleanup()


def setup_process(rank, master_addr, master_port, world_size, backend="nccl"):
    print(f"Setting up process: rank={rank} world_size={world_size} backend={backend}.")
    print(f"master_addr={master_addr} master_port={master_port}")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


if __name__=="__main__":
    for SEED in [0,1,2,3,4]:
        for DATASET_NAME in ['ms', 'pancreas', 'myeloid']:
            main(DATASET_NAME=DATASET_NAME, SEED=SEED)
