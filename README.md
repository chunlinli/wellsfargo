# Wells Fargo Campus Analytics Challenge

Team Member: Xuesong Hou, Chunlin Li, Yu Yang

This repository contains all of the code, data, and images throughout our data anlaysis. 

### Data
- `train.csv`, `test.csv`: converted from the original excel file provided by the challenge.
- `pred.csv`: predictions given by our model.

### Notebooks
- `eda.ipynb`: exploratory data analysis.
- `models.ipynb`: model comparison, encoding scheme investigation.
- `group.ipynb`: fit sparse grouping pursuit models.
- `models.py`: functions used in `models.ipynb`.
- `group.py`: functions used in `group.ipynb`.

## Environment Configuration
Create a conda environment using the following command in your shell.
```bash
conda env create -f environment.yml
```

Activate or deactivate the virtual environment using the following commands.
```
conda activate wellsfargo
conda deactivate
```
