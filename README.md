# FlareSense
ML-Powered Solar Flare Classification with e-Callisto.

Instructions
------------
1. Clone the repo.
2. Run `make reqs` to install required python packages.
3. Setup the DVC credentials using DagsHub.
4. You're ready to start developing!

Project Organization
------------

    ├── .dvc               <- Keeps the data pipeline versioned.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering)
    │                         and a short `-` delimited description, e.g.
    │                         `1.0-initial-data-exploration`.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            <- Generated notebooks, metrics, figures, and other reports.
    ├── .dvcignore         <- Files and directories to ignore by DVC.
    ├── .gitignore         <- Files and directories to ignore by Git.
    ├── dvc.yaml           <- Defining the data pipeline stages, dependencies, and outputs.
    ├── LICENSE            <- GNU General Public License v3.0.
    ├── Makefile           <- Makefile with commands.
    ├── params.yaml        <- The parameters for the data pipeline.
    ├── raw_data.yaml      <- Version control config for the raw data.
    ├── README.md          <- The top-level README for developers using this project.
    ├── dvc.lock           <- The version definition of each dependency, stage, and output from the 
    │                         data pipeline.
    └── requirements.txt   <- The requirements file for reproducing the analysis environment.