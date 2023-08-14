# Bayesian Updating of Hurricane Vulnerability Functions

## File structure

```
├── README.md
├── actual_model
│   ├── LICENSE
│   ├── estimate_posterior.R
│   ├── labelled_data
│   ├── max_wind_field.tif
│   ├── observations_bad.csv
│   ├── observations_good.csv
│   ├── observations_medium.csv
│   └── parse_observations.py
├── archive
│   ├── test
│   └── train
├── assets
│   ├── data
│   ├── posterior_parameters
│   ├── posterior_parameters_genesis
│   ├── posteriors
│   ├── posteriors_genesis
│   └── priors
│   └── results
├── src
│   ├── 0_createSimulatedData.py
│   ├── 1_model.py
│   ├── 2_DrawPosterior_main.py
│   ├── 2_DrawPosterior_test.py
│   ├── 3_contribution.ipynb
│   ├── 3_contribution.py
│   ├── 4_contribution_GPU.ipynb
│   ├── 5_zoib_regression_bayesian.ipynb
│   ├── helper
│   └── test.py
└── venv
```

- actual_model 
    contains the code of the author along with the required data files/
- archive 
    is not present in github due to large size of the folder. It contains the data for the contribution.ipynb
-assets
    contains all the other data files generated or used for model implementation, data simulation, posterior plotting
- src
    contains all the source codes. 

    - 0_createSimulatedData.py contains the code for data simulation
    - 1_model.py contains the code for model implementation
    - 2_DrawPosterior_main.py is used for drawing the final posterior plots
    - <b>3_contribution.ipynb contains the source code for our contribution for the pm project </b>
    - <b> 4_contribution_GPU.ipynb contains the same for the contribution.ipynb using the gpu </b>
    - <b> 5_zoib_regression_bayesian.ipynb contains the colab version of the model implementation and data simulation </b>
    - helper folder contains some utility files. 
- venv 
    represents the virtual environment. Also, not found in github.

## Acknowledgement

We acknowledge that this project followed the paper for reimplementation of the model and project work.

Jens A. de Bruijn et al. 'Using rapid damage observations for Bayesian updating of hurricane vulnerability functions: A case study of Hurricane Dorian using social media'. In: (2022). doi: https://doi.org/10.1016/j.ijdrr.2022.102839.