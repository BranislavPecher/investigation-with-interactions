# Investigating effects of randomness with interactions

## Prerequisites

* [Docker](https://www.docker.com/)


## Installation and running the investigation

To run the project properly with correct libraries and prerequisites, we strongly recommend using `Docker`.

To run this project, make sure you have Docker installed and follow the steps:

1. Get into the project root directory.
1. Build docker image:
    ```bash
    docker build -t randomness_investigation .
    ```
1. Prepare config files to be run (use the prepared configs to run the investigation done in our paper, or use them as examples)
1. Download and preprocess data using the config files contained in respective folders (e.g. `config/{dataset}/preprocess.json`).
1. Copy the preprocessed data to the folder specified by investigation config (or modify the config file)
1. Run the investigation through docker container:
    ```bash
    docker run -d --name investigation -v $(pwd):/project/ --entrypoint "/bin/bash" randomness_investigation:latest "-c" "python main.py -c config/sst2/protonet_golden.json"
    ```
1. Results from each run are stored in the specified folder. This is by default in `results/stability/{dataset}`.

## Evaluation

For the evaluation purposes, we provide jupyter notebook that are available in the `notebooks` folder. 

They are designed to work with the data from experiments, with the path to the results being modifiable by the `DATA_PATH` path. 

In addition, we also provide results from our experiments in the form of a pickled file in the `pickled` folder. Each notebook provides loading of these pickled files at the end.