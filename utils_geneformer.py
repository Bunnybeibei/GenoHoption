import dgl
import json
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
# from expander import get_hamilton_graph

# map vocab
def map_vocab(all_data):
    all_data['source'] = all_data['source'].map(vocab)
    all_data['target'] = all_data['target'].map(vocab)
    all_data = all_data.dropna()
    all_data = all_data.reset_index(drop=True)
    all_data['source'] = all_data['source'].astype(int)
    all_data['target'] = all_data['target'].astype(int)
    all_data['importance'] = all_data['importance'].astype(float)
    return all_data

# import vocab
######################### change vocab ################################
# Step1: Import
import os, pickle
geneformer_data = "weights/Geneformer"
dict_dir = os.path.join(geneformer_data, "dicts")
token_name_id_path = os.path.join(dict_dir, "gene_name2id_dict.pkl")
with open(token_name_id_path, "rb") as f:
    vocab = pickle.load(f)
# Step2: Add cls
vocab['<cls>'] = len(vocab)
######################### change vocab ################################

# import grn
grn_prior = pd.read_csv('data/Real_data/grn_gather.csv')
grn_prior['importance'] = 0.90
grn_prior = map_vocab(grn_prior)
grn_prior = grn_prior.drop_duplicates(subset=['source', 'target'])

def create_adj_mat(attention_window, max_len=4096, num_rand=64, num_glob=64):
    attention_window = (
        attention_window
        if isinstance(attention_window, int)
        else max(attention_window)
    )
    # max_len = 4096  # not the input sequence max len
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
    np.random.seed(0)
    num_random = max_len * num_rand

    idx = np.random.choice(range(max_len * max_len), num_random, replace=False)
    idx_x = idx % max_len
    idx_y = idx // max_len
    adj[idx_x, idx_y] = 1

    # add global att
    num_global = num_glob
    idx = np.random.choice(range(attention_window, max_len), num_global, replace=False)
    adj[idx, :] = 1
    adj[:, idx] = 1

    possible_seq_len = np.arange(attention_window, max_len + attention_window, attention_window)
    src_dst = {k: np.nonzero(adj[:k, :k]) for k in possible_seq_len}
    # 生成一个字典，可以直接提取指定长度的连接情况，一般是num_nodes * num_nodes + random_num
    return src_dst


def pad_to_window_size(attention_window, inputs, padding_value=0):
    """
    它是对输入序列做右端padding,目的是让序列长度可以被attention window整除,这样在模型中计算self-attention时,可以对整个窗口进行并行计算。
    Args:
        inputs:

    Returns:

    """
    attention_window = (
        attention_window
        if isinstance(attention_window, int)
        else max(attention_window)
    )
    assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
    input_shape = inputs.shape
    batch_size, seq_len = input_shape[:2]
    padding_len = (attention_window - seq_len % attention_window) % attention_window
    if padding_len > 0:
        # print(
        #     f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
        #     f"`config.attention_window`: {attention_window}"
        # )
        if inputs is not None:
            inputs = nn.functional.pad(inputs, (padding_value, padding_len), value=padding_value)

    return inputs


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


class generate_g(object):
    def __init__(self, DATASET_NAME,
                 src_dst='both',
                 flag_global=False,
                 flag_random=0,
                 flag_hamiliton=False):
        super(generate_g, self).__init__()
        # import co-network
        if DATASET_NAME == 'ms':
            co_prior = pd.read_csv('data/Real_data/co_gather.csv')
        elif DATASET_NAME == 'pancreas':
            co_prior = pd.read_csv('data/Real_data/co_gather_pan.csv')
        elif DATASET_NAME == 'myeloid':
            co_prior = pd.read_csv('data/Real_data/co_gather_melo.csv')
        elif DATASET_NAME == "adamson":
            co_prior = pd.read_csv('data/Real_data/co_gather_ad.csv')
        elif DATASET_NAME == "norman":
            co_prior = pd.read_csv('data/Real_data/co_gather_nor.csv')
        else:
            raise ValueError(f"Invalid dataset name: {DATASET_NAME}")
        co_prior = map_vocab(co_prior)
        self.co_prior = co_prior.drop_duplicates(subset=['source', 'target'])

        self.src_dst = src_dst
        self.flag_global = flag_global
        self.flag_random = flag_random
        self.flag_hamiliton = flag_hamiliton

    def from_cor_to_batched_graphs(self, inputs):
        B = inputs.shape[0]
        gene_id_list = inputs.detach().cpu().numpy().tolist() #(b, N)
        g_list = []

        if self.src_dst == 'both':
            src_dst = pd.merge(grn_prior[['source', 'target','importance']], \
                               self.co_prior[['source', 'target','importance']], \
                               how='outer')
        elif self.src_dst == 'co':
            src_dst = self.co_prior
        elif self.src_dst == 'grn':
            src_dst = grn_prior
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
            ##################### add hamiliton #######################################
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
            ############################################################################
            g_list.append(g)

        batched_g = dgl.batch(g_list)
        return batched_g


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    dicts = create_adj_mat(64, 2048)

    inputs = {}
    inputs['gene_ids'] = torch.randint(low=0, high=100,size=(32,2048))
    inputs['values'] = torch.randn(size=(32,2048))

    input, src_key_padding_mask = pad_to_window_size(64, inputs, pad_token_id=0, src_key_padding_mask=torch.ones(size=(32, 10)))
    g = from_adj_to_batched_graphs(dicts, inputs['gene_ids'])

    print(f'g\'s in_degrees is {g.in_degrees()}')
    print(f'g\'s out_degrees is {g.out_degrees()}')
    print('print in-degree and out-degree')

    degrees = g.in_degrees() + g.out_degrees()
    is_all_equal = torch.all(degrees == degrees[0]).item()

    if is_all_equal:
        print("All elements of the tensor are equal.")
    else:
        print("Not all elements of the tensor are equal.")
        print(f'g\'s degrees is {degrees}')

    Generator = generate_g(DATASET_NAME='ms')
    g1 = Generator.from_cor_to_batched_graphs(inputs['gene_ids'], src_dst='co', flag_random=2)
    print("Test passed!")