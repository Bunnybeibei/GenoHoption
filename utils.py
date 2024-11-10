"""
This code contains the tools needed to compare with scGPT as the backbone.
"""
import dgl
import json
import scanpy as sc
from scipy.sparse import issparse
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
# from expander import get_hamilton_graph (deprecated)
from typing import Union


############################# map vocab ################################
"""
This function maps gene names to token values.
"""
def map_vocab(all_data):
    all_data['source'] = all_data['source'].map(vocab)
    all_data['target'] = all_data['target'].map(vocab)
    all_data = all_data.dropna()
    all_data = all_data.reset_index(drop=True)
    all_data['source'] = all_data['source'].astype(int)
    all_data['target'] = all_data['target'].astype(int)
    all_data['importance'] = all_data['importance'].astype(float)
    return all_data


############################# import vocab #############################
"""
This function imports pre-trained token embeddings.
"""
with open('weights/scgpt/scGPT_human/vocab.json', 'r') as f:
    vocab = json.load(f)


############################# create adjacency matrices (Comparison) #############################
"""
This function is used to create adjacency matrices.
(Used for comparison methods, modes include window, longformer, bigbird.)
"""
def create_adj_mat(attention_window, max_len=4096, num_rand=200, mode='window'):
    attention_window = (
        attention_window
        if isinstance(attention_window, int)
        else max(attention_window)
    )
    n_blocks = max_len // (attention_window // 2) - 1
    adj = np.zeros([max_len, max_len])

    # add local window att (overlap)
    for i in range(n_blocks):
        start = i * attention_window // 2
        end = start + attention_window
        if end > max_len:
            end = max_len
        adj[start:end, start:end] = 1

    if mode == 'bigbird':
        # add random att
        # np.random.seed(0)
        num_random = max_len * num_rand

        idx = np.random.choice(range(max_len * max_len), num_random, replace=False)
        idx_x = idx % max_len
        idx_y = idx // max_len
        adj[idx_x, idx_y] = 1

        # add global att
        adj[0, :] = 1
        adj[:, 0] = 1

    possible_seq_len = np.arange(attention_window, max_len + attention_window, attention_window)
    src_dst = {k: np.nonzero(adj[:k, :k]) for k in possible_seq_len}
    return src_dst


############################# padding (Comparison) #############################
"""
This function is used for padding.
(Used for comparison methods, modes include window, longformer, bigbird.)
"""
def pad_to_window_size(attention_window, inputs, pad_token_id=36571,src_key_padding_mask=None, pad_value=0):

    attention_window = (
        attention_window
        if isinstance(attention_window, int)
        else max(attention_window)
    )
    assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
    input_shape = inputs["gene_ids"].shape if inputs["gene_ids"] is not None else src_key_padding_mask.shape
    batch_size, seq_len = input_shape[:2]
    padding_len = (attention_window - seq_len % attention_window) % attention_window
    if padding_len > 0:
        # print(
        #     f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
        #     f"`config.attention_window`: {attention_window}"
        # )
        if inputs["gene_ids"] is not None:
            inputs["gene_ids"] = nn.functional.pad(inputs["gene_ids"], (0, padding_len),
                                                    value=pad_token_id)
        if inputs["values"] is not None:
            inputs["values"] = nn.functional.pad(inputs["values"], (0, padding_len),
                                                    value=pad_value)
        src_key_padding_mask = nn.functional.pad(
            src_key_padding_mask, (0, padding_len), value=False
        )  # no attention on the padding tokens
    return inputs, src_key_padding_mask


############################# creating graphs (Comparison) #############################
"""
This function is used for creating graphs.
(Used for comparison methods, modes include window, longformer, bigbird.)
"""
def from_adj_to_batched_graphs(src_dst, input_ids):
    B = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    g_list = []
    for i in range(B):
        src, dst = src_dst[seq_len] #src and dst are ndarray format
        g = dgl.graph((src, dst))
        g_list.append(g)
    batched_g = dgl.batch(g_list)
    return batched_g


############################# creating graphs (Ours) #############################
"""
This class is used to construct our single-cell gene graph.
"""
class generate_g(object):
    def __init__(self, DATASET_NAME,
                 src_dst='both',
                 flag_global=False,
                 flag_random=0,
                 flag_hamiliton=False):
        super(generate_g, self).__init__()
        """
        Import prior knowledge.
        """

        # import grn
        grn_prior = pd.read_csv('data/Real_data/grn_gather.csv')
        grn_prior['importance'] = 0.90
        grn_prior = map_vocab(grn_prior)
        self.grn_prior = grn_prior.drop_duplicates(subset=['source', 'target'])

        # import co-network
        if DATASET_NAME == 'ms':
            co_prior = pd.read_csv('data/Real_data/co_gather_ms.csv')
        elif DATASET_NAME == 'pancreas':
            co_prior = pd.read_csv('data/Real_data/co_gather_pan.csv')
        elif DATASET_NAME == 'myeloid':
            co_prior = pd.read_csv('data/Real_data/co_gather_melo.csv')
        elif DATASET_NAME == "adamson":
            co_prior = pd.read_csv('data/Real_data/co_gather_ad.csv')
        elif DATASET_NAME == "norman":
            co_prior = pd.read_csv('data/Real_data/co_gather_nor.csv')
        else:
            co_prior = pd.read_csv(f'data/Real_data/co_gather_{DATASET_NAME}_pert.csv')
        co_prior = map_vocab(co_prior)
        self.co_prior = co_prior.drop_duplicates(subset=['source', 'target'])
        self.src_dst = src_dst
        self.flag_global = flag_global
        self.flag_random = flag_random
        self.flag_hamiliton = flag_hamiliton


    def from_cor_to_batched_graphs(self, inputs):
        """
        Construct the single-cell base gene graph.
        """
        B = inputs.shape[0]
        gene_id_list = inputs.detach().cpu().numpy().tolist() #(b, N)
        g_list = []

        if self.src_dst == 'both':
            src_dst = pd.merge(self.grn_prior[['source', 'target','importance']], \
                               self.co_prior[['source', 'target','importance']], \
                               how='outer')
        elif self.src_dst == 'co':
            src_dst = self.co_prior
        elif self.src_dst == 'grn':
            src_dst = self.grn_prior
        else:
            raise ValueError("Unknown src_dst: {}".format(self.src_dst))

        for i in range(B):
            nodes = gene_id_list[i] #(N, )
            sub_src_dst = src_dst[(src_dst['source'].isin(nodes)) & (src_dst['target'].isin(nodes))]

            edges = [(nodes.index(src), nodes.index(dst), importance) for src, dst, importance in
                     zip(sub_src_dst['source'].to_list(), sub_src_dst['target'].to_list(), sub_src_dst['importance'].to_list())]

            if self.flag_global:
                edges = edges + [(nodes.index(vocab['<cls>']), nodes.index(node_id), 1.) for node_id in nodes]
                edges = edges + [(nodes.index(node_id), nodes.index(vocab['<cls>']), 1.) for node_id in nodes]
                edges = list(set(edges))

            if self.flag_random != 0:
                edges = edges + [(nodes.index(np.random.choice(nodes)), nodes.index(np.random.choice(nodes)), 1.) for _ in range(self.flag_random * B)]
                edges = list(set(edges))

            g = dgl.DGLGraph()
            g.add_nodes(len(nodes))
            src, dst, importance_list = tuple(zip(*edges))
            g.add_edges(src, dst)
            ##################### add hamiliton (deprecated) #######################################
            if self.flag_hamiliton:
                A = g.adjacency_matrix().to_dense().detach().cpu().numpy()
                A[src, dst] = importance_list
                # print('Get Adjacency matrix')
                A = np.where(A == 0., 1e-6, A)
                # print('Introduce small value')
                g = get_hamilton_graph(graph=g, A=A)
            else:
                ###################### add self circle #################################
                g = dgl.add_self_loop(g)
                ########################################################################
            g_list.append(g)

        batched_g = dgl.batch(g_list)
        return batched_g


############################# loss calculation of the perturbation task #############################
def loss_fct(pred, y, perts, ctrl=None, direction_lambda=1e-3):
    """
    Main MSE Loss function, includes direction loss

    Args:
        pred (torch.tensor): predicted values
        y (torch.tensor): true values
        perts (list): list of perturbations
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions

    """
    gamma = 2
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)

    for p in set(perts):
        pert_idx = np.where(perts == p)[0]
        if ctrl.shape[0] != pred.shape[1]:
            pred = pred[:, :-1]
            y = y[:, :-1]

        pred_p = pred[pert_idx]
        y_p = y[pert_idx]
        losses = losses + torch.sum((pred_p - y_p) ** (2 + gamma)) / pred_p.shape[0] / pred_p.shape[1]

        losses = losses + torch.sum(direction_lambda * (torch.sign(y_p - ctrl) -
                                                        torch.sign(pred_p - ctrl)) ** 2) / \
                 pred_p.shape[0] / pred_p.shape[1]
    return losses / (len(set(perts)))


############################# set seed #############################
def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)


def map_raw_id_to_vocab_id(
    raw_ids: Union[np.ndarray, torch.Tensor],
    gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map some raw ids which are indices of the raw gene names to the indices of the

    Args:
        raw_ids: the raw ids to map
        gene_ids: the gene ids to map to
    """
    if isinstance(raw_ids, torch.Tensor):
        device = raw_ids.device
        dtype = raw_ids.dtype
        return_pt = True
        raw_ids = raw_ids.cpu().numpy()
    elif isinstance(raw_ids, np.ndarray):
        return_pt = False
        dtype = raw_ids.dtype
    else:
        raise ValueError(f"raw_ids must be either torch.Tensor or np.ndarray.")

    if raw_ids.ndim != 1:
        raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

    if gene_ids.ndim != 1:
        raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

    mapped_ids: np.ndarray = gene_ids[raw_ids]
    assert mapped_ids.shape == raw_ids.shape
    if return_pt:
        return torch.from_numpy(mapped_ids).type(dtype).to(device)
    return mapped_ids.astype(dtype)


############################# Construct a co-expression dataset #############################
def co_expression(adata,
                  select_hvg=False,
                  input_layer_key="X_binned",
                  DATASET_NAME='ms',
                  n_neighbours=20,
                  threshold=0.4
                  ):
    if select_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=None, subset=True)
    gene_list = adata.var['gene_name'].to_list()
    idx2gene = dict(zip(range(len(gene_list)), gene_list))

    def np_pearson_cor(x, y):
        xv = x - x.mean(axis=0)  # (sample_num, gene_num)
        yv = y - y.mean(axis=0)  # (sample_num, gene_num)
        xvss = (xv * xv).sum(axis=0)
        yvss = (yv * yv).sum(axis=0)
        result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
        # bound the values to -1 to 1 in the event of precision issues
        return np.maximum(np.minimum(result, 1.0), -1.0)

    n_neighbours = n_neighbours
    threshold = threshold
    out = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    out = np_pearson_cor(out, out)
    out[np.isnan(out)] = 0
    out = np.abs(out)
    out_sort_idx = np.argsort(out)[:, -(n_neighbours + 1):]
    out_sort_val = np.sort(out)[:, -(n_neighbours + 1):]
    df_g = []
    for i in range(out_sort_idx.shape[0]):
        target = idx2gene[i]
        for j in range(out_sort_idx.shape[1]):
            source = idx2gene[out_sort_idx[i, j]]
            df_g.append((source, target, out_sort_val[i, j]))

    df_g = [i for i in df_g if i[2] > threshold]
    co_prior = pd.DataFrame(df_g).rename(columns={0: 'source', 1: 'target', 2: 'importance'})
    co_prior.to_csv(f'data/Real_data/co_gather_{DATASET_NAME}_pert.csv', index=False)


if __name__ == "__main__":
    """Use example."""
    np.random.seed(0)
    torch.manual_seed(0)

    # This section of code constructs a graph for testing comparison methods.
    dicts = create_adj_mat(64, 2048)

    inputs = {}
    inputs['gene_ids'] = torch.randint(low=0, high=100,size=(32,2048))
    inputs['values'] = torch.randn(size=(32,2048))

    input, src_key_padding_mask = pad_to_window_size(64, inputs, pad_token_id=0, src_key_padding_mask=torch.ones(size=(32, 10)))
    g = from_adj_to_batched_graphs(dicts, inputs['gene_ids'])

    print(f'g\'s in_degrees is {g.in_degrees()}')
    print(f'g\'s out_degrees is {g.out_degrees()}')
    print('print in-degree and out-degree')

    # This section of code constructs a graph for testing ours.
    Generator = generate_g(DATASET_NAME='ms')
    g1 = Generator.from_cor_to_batched_graphs(inputs['gene_ids'])
    print("Test passed!")