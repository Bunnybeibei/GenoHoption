# GenoHoption 🐰
What can we do if we already have a line of single-cell LLMs and want to use the graphs to enhance gene-gene interaction? 🤔

Why not have a try on GenoHoption?

## Overview
The workflow of GenoHoption is shown below:
![Overview](https://github.com/user-attachments/assets/88703905-31d4-4a51-b120-dc1fa29c6283)


## Key Code
- Iteration
  GenoHoption/
    - Geneformer_adaptor.py (Transfomer Wrapper Version)
    - Geneformer_window.py (Transfomer Wrapper Version)
    - gn_attn.py (Clear Version of Attention by DGL)
    - gn_encoder.py (Clear Version of Attention by DGL)
    - gn_utils.py (Clear Version of Attention by DGL)
- Data Preprocess
  - utils_geneformer.py (For Geneformer as the backbone)
  - utils.py (For scGPT as the backbone)
- Seq2Graph
  class generate_g (use ''find'' to search)
## Usage
- Cell-type Annotation
  - Geneformer as Backbone
    - Geneformer_run_all_celltype_fewshot.py (backbone)
    - Geneformer_run_all_celltype_gn.py (GenoHoption)
    - Geneformer_window.py (Bigbird/Longformer/Diffuser)
  - scGPT as Backbone
    - scGPT_run_all_celltypeannot_fewshot.py (backbone)
    - scGPT_run_all_celltype_gn.py (GenoHoption)
    - scGPT_run_all_celltype_window.py (Bigbird/Longformer/Diffuser)
  - scBERT
    - dist_finetune_fewshot.py
  - For R4's demand
    - scGAD.py
- Perturbation Prediction
  - scGPT_run_pert_fewshot.py (backbone)
  - scGPT_run_pert_gn.py (GenoHoption)
- Complexity Analysis
  - Complex.py
## Installation
We strongly advise that you individually install `FlashAttention, PyTorch` on your device. Here are some configurations from our device for reference:

- CUDA == 11.7
- Python == 3.8
- flash-attn == 1.0.1
- torch == 1.13.0+cu117
- [*DGL*](https://www.dgl.ai/pages/start.html)
## Model Parameters
The parameters can be downloaded from these links:
- Geneformer: [*link*](https://huggingface.co/ctheodoris/Geneformer), 
- scGPT: [*link*](https://github.com/bowang-lab/scGPT),
- scBERT: [*link*](https://github.com/TencentAILabHealthcare/scBERT). 
## Datasets
- Cell-type Annotation
  - As provided by the scGPT authors:
    - Multiple Sclerosis (M.S.) dataset: [*link*](https://drive.google.com/drive/folders/1Qd42YNabzyr2pWt9xoY4cVMTAxsNBt4v)
    - Myeloid (Mye.) dataset: [*link*](https://drive.google.com/drive/folders/1VbpApQufZq8efFGakW3y8QDDpY9MBoDS)
    - hPancreas dataset: [*link*](https://drive.google.com/drive/folders/1s9XjcSiPC-FYV3VeHrEa7SeZetrthQVV)
- Perturbation Prediction
  - The same as [*GEARS*](https://github.com/snap-stanford/GEARS) and download by a piece of code:
    ```python
    pert_data.load(data_name=dataset_name)
    ```
- GRN
  - The dataset of the GRN network: [*link*](https://github.com/yangkaiyuan1027/DGP-AMIO/tree/main/graphs)
  - 
## Citing GenoHoption
```bibtex
@INPROCEEDINGS{10822153,
  author={Cheng, Jiabei and Li, Jiachen and Yang, Kaiyuan and Shen, Hongbin and Yuan, Ye},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={GenoHoption: Bridging Gene Network Graphs and Single-Cell Foundation Models}, 
  year={2024},
  volume={},
  number={},
  pages={1453-1456},
  keywords={Bridges;Sequential analysis;Foundation models;Biological system modeling;Message passing;Computational modeling;Perturbation methods;Predictive models;Genetics;Periodic structures;Single-cell Foundation Models;Gene Network Graphs;Graph Diffusion},
  doi={10.1109/BIBM62325.2024.10822153}}
}
```
  
