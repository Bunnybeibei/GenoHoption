#RB
import os
import torch


# %%
import copy
import gc
import json
import dgl
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import pickle
from anndata import AnnData
import scanpy as sc
import scvi
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from scgpt.utils import map_raw_id_to_vocab_id
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model.model_gn import AdversarialDiscriminator
from scgpt.model.generation_modle_gn import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
from utils import generate_g, loss_fct

from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis, non_zero_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
os.environ["WANDB_MODE"] = "offline"
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')

params = ('both', True, 0, False)
FRAC = 0.75
SEED = 0
alpha = 0.25
temperature = 5.
iter_num = 6
use_gears_func=True

for DATASET_NAME in ["norman","adamson"]:
    for diffusion_mode in [0, 1]:
        for SEED in [0,1,2,3,4]:
            print(DATASET_NAME)
            print("fraction: ", FRAC)
            print("SEED: ", SEED)

            ## Step1: Specify hyper-parameter setup for cell-type annotation task
            # Listed below are some hyper-parameter recommendations for the cell-type task. Note that the CLS objective is on to facilitate cell-type classification.

            hyperparameter_defaults = dict(
                seed=SEED,
                dataset_name=DATASET_NAME,
                fraction=FRAC,
                do_train=True,
                load_model="weights/scgpt/scGPT_human/", #RB
                mask_ratio=0.0,
                epochs=15,
                n_bins=51,
                MVC=False, # Masked value prediction for cell embedding
                ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
                dab_weight=0.0,
                lr=1e-4,
                batch_size=32,
                layer_size=128,
                nlayers=4,  # number of nn.dTransformerEncoderLayer in nn.TransformerEncoder
                nhead=4,  # number of heads in nn.MultiheadAttention
                dropout=0.2,  # dropout probability
                schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
                save_eval_interval=5,
                fast_transformer=True,
                pre_norm=False,
                amp=True,  # Automatic Mixed Precision
                include_zero_gene = "all",
                freeze = False, #freeze
                DSBN = False,  # Domain-spec batchnorm
                ################# add diffusion_mode #####################
                diffusion_mode = diffusion_mode,
                ###########################################################
                ################# add hyper params #####################
                iter_num = iter_num,
                temperature = temperature,
                alpha = alpha,
                ###########################################################
            )

            run = wandb.init(
                config=hyperparameter_defaults,
                project="Predict Perturbation",
                name=f'Trail_{DATASET_NAME}_{SEED}',
                group='heat' if diffusion_mode == 1 else 'ppr_backup',
                reinit=True,
                settings=wandb.Settings(start_method="fork"),
            )
            config = wandb.config
            print(config)

            set_seed(config.seed)

            # settings for input and preprocessing
            pad_token = "<pad>"
            pert_pad_id = 2
            use_fast_transformer = True  # whether to use fast transformer
            special_tokens = [pad_token, "<cls>", "<eoc>"]
            mask_ratio = config.mask_ratio
            mask_value = "auto"  # for masked values, now it should always be auto

            include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
            max_seq_len = 1536 # 1536
            n_bins = config.n_bins

            # input/output representation
            input_style = "binned"  # "normed_raw", "log1p", or "binned"
            output_style = "binned"  # "normed_raw", "log1p", or "binned"

            # settings for training
            MLM = True  # whether to use masked language modeling, currently it is always on.
            CLS = False  # celltype classification objective
            ADV = False  # Adversarial training for batch correction
            CCE = False  # Contrastive cell embedding objective
            MVC = config.MVC  # Masked value prediction for cell embedding
            ECS = config.ecs_thres > 0  # Elastic cell similarity objective
            DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
            load_param_prefixs = [
                "encoder",
                "value_encoder",
                "transformer_encoder",
            ]
            INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
            input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
            cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
            adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
            adv_D_delay_epochs = 0
            mvc_decoder_style = "inner product, detach"
            ecs_threshold = config.ecs_thres
            dab_weight = config.dab_weight

            explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
            do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

            per_seq_batch_sample = False

            # settings for optimizer
            lr = config.lr  # TODO: test learning rate ratio between two tasks
            lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
            batch_size = config.batch_size
            eval_batch_size = config.batch_size
            epochs = config.epochs
            schedule_interval = 1

            # settings for the model
            fast_transformer = config.fast_transformer
            fast_transformer_backend = "gn"  # "linear" or "flash" or "gn"
            embsize = config.layer_size  # embedding dimension
            d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
            nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
            nhead = config.nhead  # number of heads in nn.MultiheadAttention
            dropout = config.dropout  # dropout probability

            # logging
            log_interval = 100 # iterations
            save_eval_interval = config.save_eval_interval  # epochs
            do_eval_scib_metrics = True

            # %% validate settings
            assert input_style in ["normed_raw", "log1p", "binned"]
            assert output_style in ["normed_raw", "log1p", "binned"]
            assert input_emb_style in ["category", "continuous", "scaling"]
            if input_style == "binned":
                if input_emb_style == "scaling":
                    raise ValueError("input_emb_style `scaling` is not supported for binned input.")
            elif input_style == "log1p" or input_style == "normed_raw":
                if input_emb_style == "category":
                    raise ValueError(
                        "input_emb_style `category` is not supported for log1p or normed_raw input."
                    )

            if input_emb_style == "category":
                mask_value = n_bins + 1
                pad_value = n_bins  # for padding gene expr values
                n_input_bins = n_bins + 2
            else:
                mask_value = -1
                pad_value = -2
                n_input_bins = n_bins

            if ADV and DAB:
                raise ValueError("ADV and DAB cannot be both True.")
            DAB_separate_optim = True if DAB > 1 else False

            dataset_name = config.dataset_name
            fraction = config.fraction
            save_dir = Path(f"./save/scGPT_few_shot/ppr_backup")
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"save to {save_dir}")
            logger = scg.logger
            scg.utils.add_file_handler(logger, save_dir / "run.log")

            ## Step 2: Load and pre-process data
            #We follow the standard scGPT data pre-processing pipelines for the cell-type annotation task. Note that since now we have two datasets at hand (i.e., reference and query data), the same pre-prpocessing steps need to be applied to both of them.

            split = "simulation"
            if dataset_name == "adamson":
                perts_to_plot = ["KCTD16+ctrl"]
                pert_data = PertData("data/Real_data")
                pert_data.load(data_name=dataset_name)
                pert_data.prepare_split(split=split, seed=SEED, train_gene_set_size=FRAC)
                pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

            if dataset_name == "norman": #RB
                perts_to_plot = ["SAMD1+ZBTB1"]
                pert_data = PertData("data/Real_data")
                pert_data.load(data_name=dataset_name)
                pert_data.prepare_split(split=split, seed=SEED, train_gene_set_size=FRAC)
                pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

            if config.load_model is not None:
                model_dir = Path(config.load_model)
                model_config_file = model_dir / "args.json"
                model_file = model_dir / "best_model.pt"
                vocab_file = model_dir / "vocab.json"

                vocab = GeneVocab.from_file(vocab_file)
                # shutil.copy(vocab_file, save_dir / "vocab.json")
                for s in special_tokens:
                    if s not in vocab:
                        vocab.append_token(s)

                pert_data.adata.var["id_in_vocab"] = [
                    1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
                ]
                gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
                logger.info(
                    f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
                    f"in vocabulary of size {len(vocab)}."
                )
                genes = pert_data.adata.var["gene_name"].tolist()

                # model
                with open(model_config_file, "r") as f:
                    model_configs = json.load(f)
                logger.info(
                    f"Resume model from {model_file}, the model args will override the "
                    f"config {model_config_file}."
                )
                embsize = model_configs["embsize"]
                nhead = model_configs["nheads"]
                d_hid = model_configs["d_hid"]
                nlayers = model_configs["nlayers"]
                n_layers_cls = model_configs["n_layers_cls"]
            else:
                genes = pert_data.adata.var["gene_name"].tolist()
                vocab = Vocab(
                    VocabPybind(genes + special_tokens, None)
                )  # bidirectional lookup [gene <-> int]
            vocab.set_default_index(vocab["<pad>"])
            gene_ids = np.array(
                [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
            )
            n_genes = len(genes)

            if config.load_model is None:
                vocab = Vocab(
                    VocabPybind(genes + special_tokens, None)
                )  # bidirectional lookup [gene <-> int]
            vocab.set_default_index(vocab["<pad>"])
            gene_ids = np.array(vocab(genes), dtype=int)

            ################### append cls #################################
            gene_ids = np.append(gene_ids, vocab["<cls>"])
            ################### append cls #################################
            ##################### Construct Graph ##########################
            src_dst, flag_global, flag_random, flag_hamiliton = params
            Generator_g = generate_g(DATASET_NAME=DATASET_NAME,
                                     src_dst=src_dst,
                                     flag_global=flag_global,
                                     flag_random=flag_random,
                                     flag_hamiliton=flag_hamiliton)
            input_g_all = Generator_g.from_cor_to_batched_graphs(inputs=torch.from_numpy(gene_ids).unsqueeze(dim=0))
            #################################################################

            def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
                """
                Train the model for one epoch.
                """
                model.train()
                total_loss, total_mse = 0.0, 0.0
                start_time = time.time()

                num_batches = len(train_loader)
                for batch, batch_data in enumerate(train_loader):
                    batch_size = len(batch_data.y)
                    x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
                    ori_gene_values = x[:, 0].view(batch_size, n_genes)
                    pert_flags = x[:, 1].long().view(batch_size, n_genes)
                    target_gene_values = batch_data.y  # (batch_size, n_genes)

                    if include_zero_gene in ["all", "batch-wise"]:
                        if include_zero_gene == "all":
                            input_gene_ids = torch.arange(n_genes, dtype=torch.long)
                        else:
                            input_gene_ids = (
                                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                            )
                        # sample input_gene_id
                        if len(input_gene_ids) > max_seq_len:
                            input_gene_ids = torch.randperm(len(input_gene_ids))[
                                             :max_seq_len
                                             ]

                        if use_gears_func:
                            ctrl_expression = torch.tensor(
                                np.mean(pert_data.adata.X[pert_data.adata.obs.condition == 'ctrl'],
                                        axis=0)).reshape(-1,)
                            ctrl_expression = ctrl_expression[input_gene_ids]
                        input_values = ori_gene_values[:, input_gene_ids]
                        input_pert_flags = pert_flags[:, input_gene_ids]
                        target_values = target_gene_values[:, input_gene_ids]

                        ################# add global ####################################
                        input_gene_ids = torch.cat((input_gene_ids, torch.LongTensor([len(gene_ids) - 1])))
                        ################# add global ####################################

                        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                        input_values = torch.cat(
                            [input_values, torch.full((mapped_input_gene_ids.shape[0], 1), 0)],
                            dim=1
                        )
                        target_values = torch.cat(
                            [target_values, torch.full((mapped_input_gene_ids.shape[0], 1), 0)],
                            dim=1
                        )
                        input_pert_flags = torch.cat(
                            [input_pert_flags, torch.full((mapped_input_gene_ids.shape[0], 1), 0)],
                            dim=1
                        )

                        src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                        # src_key_padding_mask = torch.zeros_like(
                        #     input_values, dtype=torch.bool
                        # )

                        subgraph = input_g_all.subgraph(input_gene_ids)
                        input_g = dgl.batch([subgraph.clone() for _ in range(batch_size)])

                    ########################## append cls ##################################
                    masked_positions = torch.ones_like(input_values, dtype=torch.bool, device=device)  # Use all
                    masked_positions[:, -1] = False
                    ########################################################################

                    mapped_input_gene_ids = mapped_input_gene_ids.to(device)
                    input_values = input_values.to(device)
                    input_pert_flags = input_pert_flags.to(device)
                    input_g = input_g.to(device)
                    src_key_padding_mask = src_key_padding_mask.to(device)
                    target_values = target_values.to(device)
                    ctrl_expression = ctrl_expression.to(device)

                    with torch.cuda.amp.autocast(enabled=config.amp):
                        output_dict = model(
                            mapped_input_gene_ids,
                            input_values,
                            input_pert_flags,
                            input_g,
                            src_key_padding_mask=src_key_padding_mask,
                            CLS=CLS,
                            CCE=CCE,
                            MVC=MVC,
                            ECS=ECS,
                        )
                        output_values = output_dict["mlm_output"]
                        if not use_gears_func:
                            loss = loss_mse = criterion(output_values, target_values, masked_positions)
                        else:
                            loss = loss_mse = loss_fct(output_values,
                                                       target_values,
                                                       perts=batch_data.pert,
                                                       ctrl=ctrl_expression)
                        wandb.log({"train/mse": loss.item()})

                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    with warnings.catch_warnings(record=True) as w:
                        warnings.filterwarnings("always")
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            1.0,
                            error_if_nonfinite=False if scaler.is_enabled() else True,
                        )
                        if len(w) > 0:
                            logger.warning(
                                f"Found infinite gradient. This may be caused by the gradient "
                                f"scaler. The current scale is {scaler.get_scale()}. This warning "
                                "can be ignored if no longer occurs after autoscaling of the scaler."
                            )
                    scaler.step(optimizer)
                    scaler.update()

                    # torch.cuda.empty_cache()

                    total_loss += loss.item()
                    total_mse += loss_mse.item()
                    if batch % log_interval == 0 and batch > 0:
                        lr = scheduler.get_last_lr()[0]
                        ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                        cur_loss = total_loss / log_interval
                        cur_mse = total_mse / log_interval
                        # ppl = math.exp(cur_loss)
                        logger.info(
                            f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                            f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                            f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
                        )
                        total_loss = 0
                        total_mse = 0
                        start_time = time.time()
                    # break


            def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
                """
                Evaluate the model on the evaluation data.
                """
                model.eval()
                total_loss = 0.0
                total_error = 0.0

                with torch.no_grad():
                    for batch, batch_data in enumerate(val_loader):
                        batch_size = len(batch_data.y)
                        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
                        ori_gene_values = x[:, 0].view(batch_size, n_genes)
                        pert_flags = x[:, 1].long().view(batch_size, n_genes)
                        target_gene_values = batch_data.y  # (batch_size, n_genes)

                        if include_zero_gene in ["all", "batch-wise"]:
                            if include_zero_gene == "all":
                                input_gene_ids = torch.arange(n_genes, dtype=torch.long)
                            else:
                                input_gene_ids = (
                                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                                )
                            # sample input_gene_id
                            if len(input_gene_ids) > max_seq_len:
                                input_gene_ids = torch.randperm(len(input_gene_ids))[
                                                 :max_seq_len
                                                 ]

                            input_values = ori_gene_values[:, input_gene_ids]
                            input_pert_flags = pert_flags[:, input_gene_ids]
                            target_values = target_gene_values[:, input_gene_ids]

                            ################# add global ####################################
                            input_gene_ids = torch.cat((input_gene_ids, torch.LongTensor([len(gene_ids) - 1])))
                            ################# add global ####################################

                            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                            input_values = torch.cat(
                                [input_values, torch.full((mapped_input_gene_ids.shape[0], 1), 0)],
                                dim=1
                            )
                            target_values = torch.cat(
                                [target_values, torch.full((mapped_input_gene_ids.shape[0], 1), 0)],
                                dim=1
                            )
                            input_pert_flags = torch.cat(
                                [input_pert_flags, torch.full((mapped_input_gene_ids.shape[0], 1), 0)],
                                dim=1
                            )

                            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                            src_key_padding_mask = torch.zeros_like(
                                input_values, dtype=torch.bool
                            )

                            subgraph = input_g_all.subgraph(input_gene_ids)
                            input_g = dgl.batch([subgraph.clone() for _ in range(batch_size)])

                        ########################## append cls ##################################
                        masked_positions = torch.ones_like(input_values, dtype=torch.bool, device=device)  # Use all
                        masked_positions[:, -1] = False
                        ########################################################################

                        mapped_input_gene_ids = mapped_input_gene_ids.to(device)
                        input_values = input_values.to(device)
                        input_pert_flags = input_pert_flags.to(device)
                        input_g = input_g.to(device)
                        src_key_padding_mask = src_key_padding_mask.to(device)
                        target_values = target_values.to(device)

                        with torch.cuda.amp.autocast(enabled=config.amp):
                            output_dict = model(
                                mapped_input_gene_ids,
                                input_values,
                                input_pert_flags,
                                input_g,
                                src_key_padding_mask=src_key_padding_mask,
                                CLS=CLS,
                                CCE=CCE,
                                MVC=MVC,
                                ECS=ECS,
                                do_sample=True,
                            )
                            output_values = output_dict["mlm_output"]
                            loss = criterion(output_values, target_values, masked_positions)
                        total_loss += loss.item()
                        total_error += masked_relative_error(
                            output_values, target_values, masked_positions
                        ).item()
                        # break

                wandb.log(
                    {
                        "valid/mse": total_loss / len(val_loader),
                        "valid/err": total_error / len(val_loader)
                    }
                )
                return total_loss / len(val_loader), total_error / len(val_loader)


            def eval_perturb(
                    loader: DataLoader, model: TransformerGenerator, device: torch.device
            ) -> Dict:
                """
                Run model in inference mode using a given data loader
                """

                model.eval()
                model.to(device)
                pert_cat = []
                pred = []
                truth = []
                pred_de = []
                truth_de = []
                results = {}
                logvar = []

                for itr, batch in enumerate(loader):
                    pert_cat.extend(batch.pert)

                    with torch.no_grad():
                        p = model.pred_perturb(batch,
                                               ######### add cls id ##########
                                               cls_id=vocab['<cls>'],
                                               ###############################
                                               include_zero_gene=include_zero_gene,
                                               gene_ids=gene_ids,
                                               ######### total_g ##########
                                               input_g_all=input_g_all,
                                               ####################################
                                               )
                        t = batch.y
                        pred.extend(p.cpu())
                        truth.extend(t.cpu())

                        # Differentially expressed genes
                        for itr, de_idx in enumerate(batch.de_idx):
                            pred_de.append(p[itr, de_idx])
                            truth_de.append(t[itr, de_idx])

                    # break

                # all genes
                results["pert_cat"] = np.array(pert_cat)
                pred = torch.stack(pred)
                truth = torch.stack(truth)
                results["pred"] = pred.detach().cpu().numpy().astype(float)
                results["truth"] = truth.detach().cpu().numpy().astype(float)

                pred_de = torch.stack(pred_de)
                truth_de = torch.stack(truth_de)
                results["pred_de"] = pred_de.detach().cpu().numpy().astype(float)
                results["truth_de"] = truth_de.detach().cpu().numpy().astype(float)

                return results

            ## Step 3: Load the pre-trained scGPT model

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cpu")

            ntokens = len(vocab)  # size of vocabulary
            model = TransformerGenerator(
                ntokens,
                embsize,
                nhead,
                d_hid,
                nlayers,
                nlayers_cls=n_layers_cls,
                n_cls=1,
                vocab=vocab,
                dropout=dropout,
                pad_token=pad_token,
                pad_value=pad_value,
                pert_pad_id=pert_pad_id,
                do_mvc=MVC,
                cell_emb_style=cell_emb_style,
                mvc_decoder_style=mvc_decoder_style,
                use_fast_transformer=use_fast_transformer,
                fast_transformer_backend=fast_transformer_backend,
                ############### diffusion_mode ###############
                diffusion_mode=config.diffusion_mode,
                ############### hyper params #################
                alpha=config.alpha,
                temperature=config.temperature,
                iter_num=config.iter_num
                ##############################################
            )
            if load_param_prefixs is not None and config.load_model is not None:
                # only load params that start with the prefix
                model_dict = model.state_dict()
                pretrained_dict_full = torch.load(model_file)
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict_full.items()
                    if any([k.startswith(prefix) for prefix in load_param_prefixs])
                }
                ################################## adapt to pretrained model #####################
                map_dict = {'attention.self.Wqkv.weight': 'self_attn.Wqkv.weight',
                            'attention.self.Wqkv.bias': 'self_attn.Wqkv.bias',
                            'attention.output.dense.weight': 'self_attn.out_proj.weight',
                            'attention.output.dense.bias': 'self_attn.out_proj.bias',
                            'attention.output.LayerNorm.weight': 'norm1.weight',
                            'attention.output.LayerNorm.bias': 'norm1.bias',
                            'intermediate.dense.weight': 'linear1.weight',
                            'intermediate.dense.bias': 'linear1.bias',
                            'output.dense.weight': 'linear2.weight',
                            'output.dense.bias': 'linear2.bias',
                            'output.LayerNorm.weight': 'norm2.weight',
                            'output.LayerNorm.bias': 'norm2.bias',
                            }
                for i in range(12):
                    for key, value in map_dict.items():
                        pretrained_dict[f'transformer_encoder.layers.{i}.{key}'] = pretrained_dict_full[f'transformer_encoder.layers.{i}.{value}']
                        pretrained_dict.pop(f'transformer_encoder.layers.{i}.{value}')
                ###################################################################################
                for k, v in pretrained_dict.items():
                    logger.info(f"Loading params {k} with shape {v.shape}")

                # print which params are not loaded
                for k, v in model_dict.items():
                    if k not in pretrained_dict:
                        print(f"Cannot load {k} with shape {v.shape}")


                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

                del pretrained_dict, pretrained_dict_full
                gc.collect()
            elif config.load_model is not None:
                try:
                    model.load_state_dict(torch.load(model_file))
                    logger.info(f"Loading all model params from {model_file}")
                except:
                    # only load params that are in the model and match the size
                    model_dict = model.state_dict()
                    pretrained_dict = torch.load(model_file)
                    pretrained_dict = {
                        k: v
                        for k, v in pretrained_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape
                    }
                    for k, v in pretrained_dict.items():
                        logger.info(f"Loading params {k} with shape {v.shape}")
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
            model.to(device)
            # wandb.watch(model)

            if ADV:
                discriminator = AdversarialDiscriminator(
                    d_model=embsize,
                    n_cls=num_batch_types,
                ).to(device)


            criterion = masked_mse_loss
            criterion_cls = nn.CrossEntropyLoss()
            criterion_dab = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, schedule_interval, gamma=config.schedule_ratio
            )
            if DAB_separate_optim:
                optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
                scheduler_dab = torch.optim.lr_scheduler.StepLR(
                    optimizer_dab, schedule_interval, gamma=config.schedule_ratio
                )
            if ADV:
                criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
                optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
                scheduler_E = torch.optim.lr_scheduler.StepLR(
                    optimizer_E, schedule_interval, gamma=config.schedule_ratio
                )
                optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
                scheduler_D = torch.optim.lr_scheduler.StepLR(
                    optimizer_D, schedule_interval, gamma=config.schedule_ratio
                )

            scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

            ## Step 4: Finetune scGPT with task-specific objectives

            best_val_loss = float("inf")
            best_avg_bio = 0.0
            best_model = None

            patient = 0
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                train_loader = pert_data.dataloader["train_loader"]
                valid_loader = pert_data.dataloader["val_loader"]

                if config.do_train:
                    train(
                        model,
                        train_loader,
                    )
                val_loss, val_mre = evaluate(
                    model,
                    valid_loader,
                )
                elapsed = time.time() - epoch_start_time
                logger.info("-" * 89)
                logger.info(
                    f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                    f"valid loss/mse {val_loss:5.4f} |"
                )
                logger.info("-" * 89)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model)
                    best_model_epoch = epoch
                    logger.info(f"Best model with score {best_val_loss:5.4f}")
                    ##########patient mechanism############
                    patient = 0
                else:
                    patient += 1
                    ##########patient mechanism############

                scheduler.step()
                ##########patient mechanism############
                if patient >= 2:
                    break
                ##########patient mechanism############
                if DAB_separate_optim:
                    scheduler_dab.step()
                if ADV:
                    scheduler_D.step()
                    scheduler_E.step()

            print("best model epoch: ", best_model_epoch)

            ## Step 5: Inference with fine-tuned scGPT model

            test_loader = pert_data.dataloader["test_loader"]
            test_res = eval_perturb(test_loader, best_model, device)
            test_metrics, test_pert_res = compute_metrics(test_res)
            wandb.log(test_metrics)
            print(test_metrics)

            deeper_res = deeper_analysis(pert_data.adata, test_res)
            non_zero_res = non_zero_analysis(pert_data.adata, test_res)
            non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

            metrics = ["pearson_delta", "pearson_delta_de"]
            metrics_non_dropout = [
                "pearson_delta_top20_de_non_dropout",
                "pearson_top20_de_non_dropout",
            ]
            metrics_non_zero = [
                'pearson_delta_top20_de_non_zero',
                'pearson_top20_de_non_zero',
            ]
            subgroup_analysis = {}
            for name in pert_data.subgroup["test_subgroup"].keys():
                subgroup_analysis[name] = {}
                for m in metrics:
                    subgroup_analysis[name][m] = []

                for m in metrics_non_dropout:
                    subgroup_analysis[name][m] = []

                for m in metrics_non_zero:
                    subgroup_analysis[name][m] = []

            for name, pert_list in pert_data.subgroup["test_subgroup"].items():
                for pert in pert_list:
                    for m in metrics:
                        subgroup_analysis[name][m].append(deeper_res[pert][m])

                    for m in metrics_non_dropout:
                        subgroup_analysis[name][m].append(non_dropout_res[pert][m])

                    for m in metrics_non_zero:
                        subgroup_analysis[name][m].append(non_zero_res[pert][m])

            logger_result = {}
            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    logger.info("test_" + name + "_" + m + ": " + str(subgroup_analysis[name][m]))
                    logger_result["test_" + name + "_" + m] = subgroup_analysis[name][m]

            logger_result = {key: value for key, value in logger_result.items() if not np.isnan(float(value))}
            print(logger_result)
            wandb.log(logger_result)
            #
            # # save the model into the save_dir
            # torch.save(best_model.state_dict(), save_dir / "model.pt")#RB