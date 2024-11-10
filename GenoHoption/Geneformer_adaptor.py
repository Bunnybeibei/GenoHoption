import dgl
import wandb
from dgl.data.utils import load_graphs, save_graphs
from dgl import DGLGraph
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch import nn, Tensor
from GenoHoption.gn_encoder import gn_transformer_encoder, gn_layer
from transformers import BertForSequenceClassification
from utils_geneformer import from_adj_to_batched_graphs, pad_to_window_size, create_adj_mat


# Initialize Model
def get_model(model_dir, config, mode='window'):
    model = BertForSequenceClassification.from_pretrained(model_dir,
                                                          num_labels=config.num_labels,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    encoder_parameters = model.bert.encoder.named_parameters()
    nlayers = len(model.bert.encoder.layer)

    if mode == 'window':
        config.iter_num=6
        config.alpha=0.25
    elif (mode == 'Longformer') or (mode == 'bigbird'):
        config.iter_num=1
        config.alpha=0.

    model.bert.encoder = gn_transformer_encoder(gn_layer, config, nlayers, gradient_checkpointing=False)

    # Load Pretrained
    model_encoder_dict = model.bert.encoder.state_dict()
    new_weights = []
    new_bias = []
    for name, params in encoder_parameters:
        if name in model_encoder_dict.keys():
            model_encoder_dict[name] = params
        else:
            print(f"Trying collect {name}")
            if name.split('.')[-1] == 'weight':
                new_weights.append(params)
            else:
                new_bias.append(params)
        if len(new_weights) == 3 and len(new_bias) == 3:
            i = name.split('.')[1]
            new_weights = torch.cat(new_weights, dim=0)
            new_bias = torch.cat(new_bias, dim=0)
            model_encoder_dict[f'layer.{i}.attention.self.Wqkv.weight'] = new_weights
            model_encoder_dict[f'layer.{i}.attention.self.Wqkv.bias'] = new_bias
            new_weights = []
            new_bias = []
            print(f"Layer{i} collecting is finished!")
    model.bert.encoder.load_state_dict(model_encoder_dict)
    return model


class Geneformer_adaptor(nn.Module):
    def __init__(self, config, original_model):
        super().__init__()
        self.num_labels = config.num_labels

        self.embeddings = original_model.bert.embeddings
        self.encoder = original_model.bert.encoder
        self.pooler = nn.Sequential(
            nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                input_ids: Tensor, # src
                ## add graph ##
                input_g: DGLGraph,
                ## add graph ##
                attention_mask: Tensor, # src_key_padding_mask
                labels=None,
                token_type_ids=None
                ):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        bs = embedding_output.shape[0]
        device = embedding_output.device
        if embedding_output.shape[1] * embedding_output.shape[0] != input_g.num_nodes():
            embedding_output = torch.cat([torch.mean(embedding_output,dim=1).unsqueeze(dim=1),embedding_output],dim=1)

        if attention_mask.shape[1] * attention_mask.shape[0] != input_g.num_nodes():
            attention_mask = torch.cat([torch.tensor([[0]] * bs, device=device), attention_mask],dim=1)

        # print(f'input_id_shape is {embedding_output.shape}')
        # print(f'input_g_shape is {input_g.num_nodes()}')
        # print(f'attention_shape is {attention_mask.shape}')
        ############ encoder embedding ####################
        encoder_outputs = self.encoder(
            embedding_output,
            g=input_g,
            attention_mask=attention_mask,
        )
        ############ encoder embedding ####################

        ############ pooler ####################
        sequence_output = encoder_outputs[:,0,:]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        pooled_output = self.dropout(pooled_output)
        ############ pooler ####################

        logits = self.classifier(pooled_output)

        return logits


class SeqDataset(Dataset):
    def __init__(self,
                 dataset,
                 dataset_name,
                 dataset_type,
                 indices=None,
                 graph=None,
                 ref_list=None,
                 ):
        self.dataset_dir = f"data/Real_data/{dataset_name}/Geneforme_{dataset_name}_{dataset_type}.bin"
        self.data = {}
        self.data['input_ids'] = dataset[:]['input_ids']
        self.data['label'] = dataset[:]['label']
        self.data['adata_order'] = dataset[:]['adata_order']
        self.data['length'] = dataset[:]['length']
        self.indices = indices
        self.data['gene_graph'] = []
        # graph_list = []
        # for i in range(len(self.data['input_ids'])):
        #     index = [0] + [int(ref_list.index(j)+1) for j in self.data['input_ids'][i]]
        #     graph_list.append(graph.subgraph(index))
        #     if i % 1000 == 0 and i > 0:
        #         print('1000 checkpoint')
        #         graph_labels = {"glabel": torch.tensor([range(len(graph_list))])}
        #         save_graphs(self.dataset_dir, graph_list, graph_labels)
        # graph_labels = {"glabel": torch.tensor([range(len(graph_list))])}
        # save_graphs(self.dataset_dir, graph_list, graph_labels)
        # graph_list, label_dict = load_graphs(self.dataset_dir)
        # self.data['gene_graph'] = dgl.batch(graph_list)

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        ###################### adapted graph #######################################
        result = {}
        for k, v in self.data.items():
            if k != 'gene_graph':
                result[k] = v[idx]
            else:
                # result[k] = dgl.slice_batch(v, idx)
                if self.indices is not None:
                    result[k] = load_graphs(self.dataset_dir, [self.indices[idx]])[0][0]
                else:
                    result[k] = load_graphs(self.dataset_dir, [idx])[0][0]
        ############################################################################
        if len(result['input_ids'])+1 != result['gene_graph'].num_nodes():
            print(idx)
            raise ValueError
        return result

def collate_fn(batch):
    result = {}
    temp = [torch.Tensor(item['input_ids']) for item in batch] # add cls
    result['label'] = torch.Tensor([item['label'] for item in batch])
    result['adata_order'] = [item['adata_order'] for item in batch]
    result['length'] = torch.Tensor([item['length'] for item in batch])

    result['input_ids'] = pad_sequence(temp, batch_first=True, padding_value=0)

    max_length = result['input_ids'].shape[-1] + 1
    temp = []
    for item in batch:
        a = item['gene_graph']
        if max_length-a.num_nodes() > 0:
            a.add_nodes(max_length-a.num_nodes())
        temp.append(a)
    result['gene_graph'] = dgl.batch(temp)

    return result['input_ids'].long(), result['label'].long() ,result['adata_order'],result['length'],result['gene_graph']