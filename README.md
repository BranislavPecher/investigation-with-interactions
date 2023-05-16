# On Investigation of Stability in Learning with Limited Labelled Data for Text Classification: Dealing with Interactions Between Randomness Factors

## Dependencies and local setup

The code in this repository uses Python. The required dependencies are specified in the `requirements.txt`. 

To run the project properly, with correct libraries and prerequisities, we strongly recommend using `Docker`. We provide a separate `Dockerfile` to accomplish the whole setup. To use it, follow these steps:

1. Modify the source docker image according to your need (changing CUDA versions or even foregoing GPU). The default source image is `pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime`. Note that the specified PyTorch version needs to be kept, due to requirements from other used libraries.
1. Build docker image:
    ```bash
    docker build -t randomness_investigation .
    ```

If not opting for the Docker, simply run `pip install -r requiremets.txt`.

## Running the investigation

The investigation experiments consists of multiple randomness factors (data split, label selection, model initialisation, data order, adaptation data choice and golden model), multiple dataset (SST-2, CoLA, MRPC) and multiple models (MAML, Reptile, Prototypical Networks, BERT fine-tuning), each running separately. We provide configurations files, along with the hyperparameter setup, for every experiment run in our paper in the `config` folder using following structure `config/{dataset}/{model}_{factor}.json`.

To run the specific experiment, follow these steps:

1. Prepare configuration files to be run (use either the prepared configs to run the investigation done on our paper, or use them as examples to run a separate investigation).
1. Download and preprocess data using the config file contained in respective folders (e.g. `config/{dataset}/preprocess.json`). Note that we use HuggingFace to download datasets used in our experiments.
1. Copy the preprocessed data to the folder specified by the investigation configuration file (or modify this file to point to the folder with preprocessed data).
1. Run the investigation through docker container using the configuration. Use following command (with SST-2 dataset, Prototypical Networks and Golden Model as example for the path):
    ```bash
    docker run -d --name investigation -v $(pwd):/project/ --entrypoint "/bin/bash" randomness_investigation:latest "-c" "python main.py -c config/sst2/protonet_golden.json"
    ```
1. Results from each run are stored in the specified folder. This is by default in `results/stability/{dataset}`.

The investigation of multiple models can be run at the same time, by specifying multiple models in the configuration files. However, the underlying code is setup in such a way, as to guarantee the use of same parameters (split of data, initialisation of the models, order of data, etc.) even when running the models separately, as long as the results are saved to the same directory.

If not opting for the docker, simply replace the docker command with following command `python main.py -c config/sst2/protonet_golden.json`

## Evaluating the experiments

For the evaluation purposes, we provide jupyter notebooks that are available in the `notebooks` folder. 

They are designed to work with the data from experiments, with the path to the results being modifiable by the `DATA_PATH` path. 

## Results from our experiments and pre-trained models

We provide the results from our experiments in the form of a pickled file in the `pickled` folder. Each evaluation notebook provides loading of these pickled files at the end.

The full setup of the experiment (random seeds, data splits, initialised models, orders of data, etc.) and the pre-trained models are not provided at this time due to their size.