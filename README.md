# SPARC

Code for **SPARC: Spectral graph learning for robust multimodal recommendation**,
built on [MMRec](https://github.com/enoche/MMRec) and
[MAC](https://github.com/MarineRoboticsGroup/mac).

## Installation

Use Linux with an NVIDIA GPU and Python 3.8. Run the following commands
from this repository's root directory:

```bash
conda create -n sparc python=3.8 -y
conda activate sparc
python -m pip install -r requirements.txt
python -m pip install --no-deps -e ./mac
```

The dependency file targets PyTorch 1.11.0 and CUDA 11.3. For other CUDA
versions, consult the [official PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

## Data and graphs

Prepare MMRec-format data for Baby, Clothing, or Sports. Available data
links are listed in [data/README.md](data/README.md). For each dataset, place
the interaction file, pre-extracted features, and matching unnormalized
MAC graphs in `data/<dataset>/`:

```text
data/baby/
  baby.inter
  image_feat.npy
  text_feat.npy
  image_sim_matrix_mac_k<K>_fk<F>.pt
  text_sim_matrix_mac_k<K>_fk<F>.pt
```

`K` and `F` must match `mac_knn_k` and `mac_fixed_k` in
`src/configs/model/SPARC.yaml`. Feature rows, graph nodes, and item IDs
must use the same ordering. Training loads the saved MAC graphs; it does
not generate them from features automatically.

`mac/sim2mac_fixed.py` is the retained offline sparsification utility. It
takes a precomputed dense similarity tensor as input. Use its `--help`
option for arguments; recreating archived graph files is a separate step
from training with those files.

## Training

Run from the repository root:

```bash
python src/main.py --model SPARC --dataset baby
```

Replace `baby` with `clothing` or `sports`. Model settings are in
`src/configs/model/SPARC.yaml`; data settings are in
`src/configs/dataset/`. Match the configuration to the supplied graph
files before training.

## Acknowledgments and licenses

The recommendation framework is based on MMRec; its GPLv3 license is
preserved in [LICENSE](LICENSE). The bundled MAC implementation retains
its [MIT license](mac/LICENSE). The default MAC backend is `tracemin_lu`.
The optional `tracemin_cholesky` backend additionally requires SuiteSparse
and `scikit-sparse`.
