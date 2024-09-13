# GTAM: A Molecular Pretraining Model with Geometric Triangle Awareness


Xiaoyang Hou<sup>+</sup>, Tian Zhu<sup>+</sup>, Milong Ren<sup>+</sup>, Dongbo Bu, Xin Gao, Shiwei Sun

<sup>+</sup> Equal contribution


- GTAM includes two components:
    - Contrastive learning
    - Generative learning:
        - One 2D->3D diffusion model. Frame-based SE(3)-equivariant and reflection anti-symmetric model
        - One 3D->2D diffusion model. SE(3)-invariant.



## Environments
```bash
conda create -n Geom3D python=3.7
conda activate Geom3D
conda install -y -c rdkit rdkit
conda install -y numpy networkx scikit-learn
conda install -y -c conda-forge -c pytorch pytorch=1.9.1
conda install -y -c pyg -c conda-forge pyg=2.0.2
pip install ogb==1.2.1

pip install sympy

pip install ase  # for SchNet

pip intall -e .
```

## Datasets

- For PCQM4Mv2 (pretraining) dataset
  - Download the dataset from [PCQM4Mv2 website](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) under folder `data/PCQM4Mv2/raw`:
    ```
      .
    ├── data
    │   └── PCQM4Mv2
    │       └── raw
    │           ├── data.csv
    │           ├── data.csv.gz
    │           ├── pcqm4m-v2-train.sdf
    │           └── pcqm4m-v2-train.sdf.tar.gz
    ```
  - Then run `examples/generate_PCQM4Mv2.py`.
- For QM9, it is automatically downloaded in pyg class. The default path is `data/molecule_datasets/QM9`.
- For MD17, it is automatically downloaded in pyg class. The default path is `data/MD17`.
- For MoleculeNet, please follow [GraphMVP instructions](https://github.com/chao1224/GraphMVP). The dataset structure is:
  ```
    .
  ├── data
  │   ├── molecule_datasets
  │   │   ├── bace
  │   │   │   ├── BACE_README
  │   │   │   └── raw
  │   │   │       └── bace.csv
  │   │   ├── bbbp
  ...............
  ```

## Pretraining

A quick demo on pretraining is:
```
cd examples

python pretrain.py \
--verbose --input_data_dir=../data --dataset=PCQM4Mv2 \
--lr=1e-4 --epochs=50 --num_workers=0 --batch_size=256 --SSL_masking_ratio=0 --gnn_3d_lr_scale=0.1 --dropout_ratio=0 --graph_pooling=mean --emb_dim=256 --epochs=1 \
--SDE_coeff_contrastive=1 --CL_similarity_metric=EBM_node_dot_prod --T=0.1 --normalize --SDE_coeff_contrastive_skip_epochs=0 \
--SDE_coeff_generative_2Dto3D=1 --SDE_2Dto3D_model=SDEModel2Dto3D_02 --use_extend_graph \
--SDE_coeff_generative_3Dto2D=1 --SDE_3Dto2D_model=SDEModel3Dto2D_node_adj_dense --noise_on_one_hot \
--output_model_dir=[MODEL_DIR]
```

**Notice** that the `[MODEL_DIR]` is where you are going to save your models/checkpoints.

## Downstream

The downstream scripts can be found under the `examples` folder. Below we illustrate few simple examples.
- `finetune_MoleculeNet.py`:
  ```
  python finetune_MoleculeNet.py \
  --dataset=tox21 \
  --input_model_file=[MODEL_DIR]/checkpoint/model_complete.pth
  --output_model_dir=[MODEL_DIR]/output/test/MoleculeNet/bace
  ```
- `finetune_QM9.py`: 
  ```
  python finetune_QM9.py \
  --dataset=QM9 --task=gap \
  --input_model_file=[MODEL_DIR]/checkpoint/model_complete.pth
  --output_model_dir=[MODEL_DIR]/output/test/QM9/gap
  ```
- `finetune_MD17.py`: 
  ```
  python finetune_MD17.py \
  --dataset=MD17 --task=aspirin \
  --input_model_file=[MODEL_DIR]/model_complete.pth
  --output_model_dir=[MODEL_DIR]/output/test/MD17/aspirin
  ```

# Acknowledgement
We acknowledge that the part of the code is adapted from MoleculeSDE and CarbonDesign. Thanks to the authors for sharing their code.
