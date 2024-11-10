"""
You can directly run this code. Its default mode is Ours-p.
"""
import numpy as np
import math
import dgl
from dgl.nn.functional import edge_softmax
import dgl.function as fn
import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import generate_g
from pathlib import Path
import scanpy as sc
from scgpt.tokenizer.gene_tokenizer import GeneVocab

from einops import rearrange
from performer_pytorch import FastAttention
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
import time

############################## fix seed #######################################
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


############################## Standard Attention ###############################
"""
This is the standard attention used to simulate Geneformer's attention.
"""
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(dim=1).repeat(1, 8, 1, 1) == 0, -10000)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def mask_attention_score(edges):
    # Whether an edge has feature 1
    # return (edges.data['h'] == 1.).squeeze(1)
    edge_mask = (edges.src['mask'] * edges.dst['mask']).unsqueeze(-1)  # E,1
    # edges.data['score']: [E,H,1]

    return {"score": edges.data['score'].masked_fill_(edge_mask == False, -10000)}


############################## FlashAttention ###############################
"""
This is the FlashAttention used to simulate scGPT's attention.
"""
class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                          device=qkv.device)
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                             indices, batch_size, seqlen),
                                   'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # qkv = self.Wqkv(x)
        # qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        ####################### watch memory #########################################
        torch.cuda.synchronize()
        # mem_before = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated()
        start_time = time.time()
        ##############################################################################
        context, attn_weights = self.inner_attn(x, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal)
        ####################### watch memory #########################################
        torch.cuda.synchronize()
        end_time = time.time()
        # mem_after = torch.cuda.memory_allocated()
        max_mem = torch.cuda.max_memory_allocated()
        print(f"{seq_length} 's time is {end_time - start_time}s")
        print(f'GPU memory used by method: {max_mem / (1024 ** 3)} GB')
        torch.cuda.empty_cache()
        ##############################################################################
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights, end_time - start_time, max_mem


# GenoHoption
############################ construct single-cell graph ############################
append_cls = True

model_dir = Path("weights/scgpt/scGPT_human")
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"

vocab = GeneVocab.from_file(vocab_file)
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

src_dst, flag_global, flag_random, flag_hamiliton = ('both', True, 0, False)

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
data_is_raw = False
filter_gene_by_counts = False
adata_test_raw = adata_test.copy()
adata = adata.concatenate(adata_test, batch_key="str_batch")

genes = adata.var["gene_name"].tolist()
gene_ids = np.array(vocab(genes), dtype=int)
gene_ids_add_global = np.insert(gene_ids, 0, vocab['<cls>'])
Generator_g = generate_g(DATASET_NAME='ms',
                         src_dst=src_dst,
                         flag_global=(flag_global and append_cls),
                         flag_random=flag_random,
                         flag_hamiliton=flag_hamiliton)
input_g_all = Generator_g.from_cor_to_batched_graphs(inputs=torch.from_numpy(gene_ids_add_global).unsqueeze(dim=0))
####################################################################################


if __name__ == '__main__':
    """
    !! Please pay attention that this test code calculates **the change in memory**, 
    so it requires that **no other programs** are running on the device. !!
    """
    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyper-params
    Method = 'Ours-h' # ['Performer','Standard Attention','FlashAttention','Ours-p','Ours-h']
    nheads = 8
    embed_dim = 256
    dim = embed_dim // nheads
    batch_size = 1 # batch size is 1 for Performer and 32 for other methods

    if Method != 'Performer':
        torch.set_default_dtype(torch.float16) # Since Performer dosen't support float16
        batch_size = 32 # Since Performer dosen't support batch_size >= 1 within single gpu

    for seq_length in [100, 500, 1000, 1500, 2000, 2500, 3000]:
        total_Time = 0.
        total_memory = 0.
        for _ in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:

            # attention_mask
            attention_mask = torch.ones(size=(batch_size, seq_length,))
            A_expanded = attention_mask.unsqueeze(2)
            matrix = A_expanded * attention_mask.view(batch_size, 1, seq_length)

            # simulate qkv
            padding_len = (100 - seq_length % 100) % 100
            query_vectors = torch.randn(batch_size, seq_length + padding_len, nheads, dim).to(device)  # (b,n,h,d)
            key_vectors = torch.randn(batch_size, seq_length + padding_len, nheads, dim).to(device)  # (b,n,h,d)
            value_vectors = torch.randn(batch_size, seq_length + padding_len, nheads, dim).to(device)  # (b,n,h,d)

            if Method == 'FlashAttention':
                input_vectors = torch.cat([query_vectors.unsqueeze(dim=2), \
                                           key_vectors.unsqueeze(dim=2), \
                                           value_vectors.unsqueeze(dim=2)], \
                                          dim=2)
                flash = FlashMHA(
                    embed_dim=embed_dim,
                    num_heads=nheads,
                    batch_first=True,
                    attention_dropout=0.,
                ).to(device)
                a, _, time_duration, memory_consume = flash(input_vectors, key_padding_mask=attention_mask.to(device))
                total_Time = total_Time + time_duration
                total_memory = total_memory + memory_consume / (1024 ** 3)

            elif Method == 'Performer':
                attn_fn = FastAttention(
                    dim_heads=embed_dim // nheads,
                    nb_features=None,
                    causal=False,
                    no_projection=True,
                ).to(device)
                out = attn_fn(query_vectors.permute(0, 2, 1, 3), \
                              key_vectors.permute(0, 2, 1, 3), \
                              value_vectors.permute(0, 2, 1, 3), \
                              output_attentions=True)
                torch.cuda.empty_cache()

            elif Method == 'Standard':
                ####################### watch memory #########################################
                torch.cuda.synchronize(device)
                torch.cuda.reset_max_memory_allocated()
                start_time = time.time()
                ##############################################################################
                update_v, attention_standard = attention(query_vectors.transpose(1, 2), key_vectors.transpose(1, 2),
                                                         value_vectors.transpose(1, 2), mask=matrix.to(device))
                ####################### watch memory #########################################
                torch.cuda.synchronize(device)
                end_time = time.time()
                max_mem = torch.cuda.max_memory_allocated()
                execution_time = end_time - start_time
                total_Time = total_Time + execution_time
                total_memory = max_mem / (1024 ** 3) + total_memory
                print(f"{seq_length} 's time is {execution_time}s")
                print(f'GPU memory used by method: {max_mem / (1024 ** 3)} GB')
                ##############################################################################
                update_v_standard = update_v[0].detach().cpu().numpy()
                attention_batch_0_standard = attention_standard[0, 0, :, :].detach().cpu().numpy()
                torch.cuda.empty_cache()

            else:

                g = dgl.batch(
                    [input_g_all.subgraph(torch.Tensor([i for i in range(1, seq_length + 1)]).long()) for _ in
                     range(batch_size)])
                g = g.to(device)

                # message passing
                bool_mask = (attention_mask > 0).to(device)
                g = g.local_var()
                g.ndata["mask"] = bool_mask.reshape(-1).unsqueeze(-1)  # BN,1  #padding mask #batch_size=2
                ########################## put in preprocess process ##########################
                g.ndata["virtual edge"] = bool_mask.reshape(-1).unsqueeze(-1)  # BN,1  #padding mask #batch_size=2
                #####################################################################################

                query_vectors = query_vectors.transpose(0, 1)
                key_vectors = key_vectors.transpose(0, 1)
                value_vectors = value_vectors.transpose(0, 1)
                query_vectors /= math.sqrt(dim)
                g.ndata['q'] = query_vectors.reshape(-1, nheads, dim)  # BN,H,D
                g.ndata['k'] = key_vectors.reshape(-1, nheads, dim)  # BN,H,D
                g.ndata['v'] = value_vectors.reshape(-1, nheads, dim)  # BN,H,D
                ####################### watch memory #########################################
                torch.cuda.synchronize()
                # mem_before = torch.cuda.memory_allocated()
                torch.cuda.reset_max_memory_allocated()
                start_time = time.time()
                ##############################################################################
                g.apply_edges(fn.u_dot_v('k', 'q', 'score'))  # score: [E,H,1]
                g.apply_edges(mask_attention_score)  # kq
                e = g.edata.pop('score')
                g.edata['score'] = edge_softmax(g, e)

                if Method == 'Ours-p':
                    alpha = 0.25
                    iter_num = 6

                    g.ndata["h"] = g.ndata["v"]  # \mathcal{V}_{(0)} = V = XW_{v}

                    for _ in range(iter_num):
                        g.update_all(fn.u_mul_e('h', 'score', 'm'), fn.sum('m', 'h'))
                        g.apply_nodes(lambda nodes: {'h': (1.0 - alpha) * nodes.data['h'] + alpha * nodes.data['v']})
                    # \mathcal{V}_{(k+1)}=(1-\alpha)A\mathcal{V}_{k}+\alpha V
                    ####################### watch memory #########################################
                    torch.cuda.synchronize(device)
                    end_time = time.time()
                    max_mem = torch.cuda.max_memory_allocated()
                    total_memory = max_mem / (1024 ** 3) + total_memory
                    total_Time = total_Time + end_time - start_time
                    print(f"{seq_length} 's time is {end_time - start_time}s")
                    print(f'GPU memory used by method: {max_mem / (1024 ** 3)} GB')
                    torch.cuda.empty_cache()
                    ##############################################################################

                else:
                    temperature = 5.
                    iter_num = 6

                    g.ndata["h"] = math.exp(-temperature) * g.ndata["v"]
                    gather_tensor = g.ndata["h"]  # [\mathcal{V}_{(0)} = V = e^{-t}XW_{v}]
                    # akv = g.ndata['h']  # since h is updating
                    t_power_k_fact = 1.

                    for k in range(1, iter_num):
                        g.update_all(fn.u_mul_e('h', 'score', 'm'), fn.sum('m', 'h'))  # A \cdot \mathcal{V}_{k}
                        # compute t^k / k!
                        t_power_k_fact *= temperature / k
                        # accumulate
                        gather_tensor = gather_tensor + t_power_k_fact * g.ndata['h']
                        # \mathcal{V}_{(k+1)}= \dfrac{tA}{k+1}\mathcal{V}_{k}

                    g.apply_nodes(lambda nodes: {'h': gather_tensor})
                    # \sum_{k=0}^{\infty}\mathcal{V}_{(k)}=(I+tA+\dfrac{t^{2}A^{2}}{2}+...)e^{-t}V
                    ####################### watch memory #########################################
                    torch.cuda.synchronize(device)
                    end_time = time.time()
                    max_mem = torch.cuda.max_memory_allocated()
                    total_memory = max_mem / (1024 ** 3) + total_memory
                    total_Time = total_Time + end_time - start_time
                    print(f"{seq_length} 's time is {end_time - start_time}s")
                    print(f'GPU memory used by method: {max_mem / (1024 ** 3)} GB')
                    torch.cuda.empty_cache()
                    ##############################################################################

        print(f'{seq_length} s: {total_Time / 10}')
        print(f'{seq_length} GB: {total_memory / 10}')