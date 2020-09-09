# `MAGIC`
**MAGIC**, Multi-scAle heteroGeneity analysIs and Clustering, is a semi-supervised clustering method that combines orthogonal projective non-negative matrix factorization and [pyHYDRA](https://github.com/anbai106/pyHYDRA) through double cyclic optimization for more pathologically plausible and robust clustering solutions for brain diseases.

Compared to original HYDRA method, MAGIC has the following advantages:
- Data-driven fashion for multi-scale feature extraction via OPNMF;
- Demonstration with semi-simulation experiments for better clustering performance.

## Installation
### Prerequisites
In order to run MAGIC, one must have installed [sopNMF]() and [pyHYDRA](https://github.com/anbai106/pyHYDRA). After this, please follow the following steps for isntallation.

There are three choices to install MAGIC.
### Use MAGIC as a python package (TODO)
We recommend the users to use Conda virtual environment:
```
1) conda create --name MAGIC python=3.6
```
Activate the virtual environment:
```
2) source activate MAGIC
```
Install other python package dependencies (go to the root folder of MAGIC):
```
3) ./install_requirements.sh
```
Finally, we need install MAGIC from PyPi:
```
3) pip install magic==1.0.0
```

### Use MAGIC from commandline (TODO):
After installing all dependencies in the **requirements.txt** file, go to the root folder of MAGIC where the **setup.py** locates:
```
pip install -e .
```

### Use MAGIC as a developer version:
```
python -m pip install git+https://github.com/anbai106/MAGIC.git
```

## Input structure
MAGIC requires a specific input structure inspired by [BIDS](https://bids.neuroimaging.io/).
Some conventions for the group label/diagnosis: -1 represents healthy control (**CN**) and 1 represents patient (**PT**); categorical variables, such as sex, should be encoded to numbers: Female for 0 and Male for 1, for instance.

### participant and covariate tsv
The first 3 columns are **participant_id**, **session_id** and **diagnosis**.

Example for feature tsv:
```
participant_id    session_id    diagnosis 
sub-CLNC0001      ses-M00    -1   432.1
sub-CLNC0002      ses-M00    1    398.2
sub-CLNC0003      ses-M00    -1    412.0
sub-CLNC0004      ses-M00    -1    487.4
sub-CLNC0005      ses-M00    1    346.5
sub-CLNC0006      ses-M00    1    443.2
sub-CLNC0007      ses-M00    -1    450.2
sub-CLNC0008      ses-M00    1    443.2
```
Example for covariate tsv:
```
participant_id    session_id    diagnosis    age    sex ...
sub-CLNC0001      ses-M00    -1   56.1    0
sub-CLNC0002      ses-M00    1    57.2    0
sub-CLNC0003      ses-M00    -1    43.0    1
sub-CLNC0004      ses-M00    -1    25.4    1
sub-CLNC0005      ses-M00    1    74.5    1
sub-CLNC0006      ses-M00    1    44.2    0
sub-CLNC0007      ses-M00    -1    40.2    0
sub-CLNC0008      ses-M00    1    43.2    1
```

## Example
We offer a fake dataset in the folder of **MAGIC/data** (the pipeline is not going to run, just to show how the tsv looks like).

### Running MAGIC for clustering CN vs Subtype1 vs Subtype2 vs ...:
```
from from magic.magic_clustering import clustering
participant_tsv="MAGIC/data/participant.tsv"
opnmf_dir = "PATH_OPNMF_DIR"
output_dir = "PATH_OUTPUT_DIR"
k_min=2
k_max=8
cv_repetition=100
clustering(participant_tsv, opnmf_dir, output_dir, k_min, k_max, 25, 60, 5, cv_repetition)
```
Note that the above example assume that the input features have been corrected by covariate effects, such as age and sex.

## Citing this work
### If you use this software, please cite the following paper:
> Wen, J., Varol, E., Chand, G., Sotiras, A. and Davatzikos, C., 2020. **MAGIC: Multi-scale Heterogeneity Analysis and Clustering for Brain Diseases**. arXiv preprint arXiv:2007.00812. MICCAI 2020. - [Preprint](https://arxiv.org/abs/2007.00812)