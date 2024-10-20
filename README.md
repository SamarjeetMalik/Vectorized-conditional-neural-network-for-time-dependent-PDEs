<p align="center">
  <p align="center">
   <h1 align="center">Vectorized Conditional Neural Network for Time dependent PDEs</h1> 
  </p>

#


## Usage

The following example shows how to use the model.

```python
import torch
from vcnef.vcnef_1d import VCNeFModel as VCNeF1DModel
from vcnef.vcnef_2d import VCNeFModel as VCNeF2DModel
from vcnef.vcnef_3d import VCNeFModel as VCNeF3DModel

model = VCNeF2DModel(num_channels=4,
                     condition_on_pde_param=True,
                     pde_param_dim=2,
                     d_model=256,
                     n_heads=8,
                     n_transformer_blocks=1,
                     n_modulation_blocks=6)

# Random data with shape b, s_x, s_y, c
x = torch.rand(4, 64, 64, 4)
grid = torch.rand(4, 64, 64, 2)
pde_param = torch.rand(4, 2)
t = torch.arange(1, 21).repeat(4, 1) / 20

y_hat = model(x, grid, pde_param, t)
```

## Files
Below is a listing of the directory structure of VCNeF.

``examples.py``: Contains lightweight examples of how to use VCNeF. \
``examples_pde_bench.py``: Contains examples of how to use VCNeF with PDEBench data and the PDEBench training loop. \
``📂 vcnef``: Contains the code for the VCNeF model. \
``📂 utils``: Contains utils for the PDEBench example.


## Dataset for PDEBench Example

To use the PDEBench example ``examples_pde_bench.py``, you have to download the [PDEBench datasets](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986). An overview of the avaiable data and how to download it can be found in the [PDEBench repository](https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download). To use the downloaded datasets in the example, you have to adapt the path in ``base_path`` and the file name(s) in ``file_names``.


## Architecture
The following illustation shows the architecture of the VCNeF model for solving 2D time-dependent PDEs (e.g., Navier-Stokes equations).

![VCNeF Architecrture](img/vcnef_architecture.svg)


