# Investigating effects of randomness factor on the stability of the results, while taking interactions between randomness factors into consideration

## Dependencies and local setup

The code in this repository uses Python. The required dependencies are specified in the `requirements.txt`. 

Simply run `pip install -r requiremets.txt`.

## Running the investigation

To run the specific experiment, follow these steps:

1. Install the requirements.
1. Choose randomness factor to investigate its effects on the stability/variability of the results -- currently the repository allows to choose only a single investigated factor. The allowed parameters are: `golden_model` (special general factor), `data_split`, `label_choice`, `sample_choice`, `sampler_order`, `model_initialisation`.
1. Choose number of investigation and mitigation runs for the experiment -- the mitigation will be automatically done for all the non-investigated factors to reduce their contribution to the overall variability. This choice affects the running time (computation costs) and precision/reliability of results -- refer to our paper for best practices on how to choose these number. 
1. Choose dataset to run the investigation on. Currently we support following options: "sst2", "mrpc", "cola", "boolq", "rte". However, as we use the HuggingFace, the set of datasets can be easily extended to include other ones (the dataset classes in `data.py` file needs to be extended with the loading and processing of the new dataset). 
1. Run the investigation using following command (with SST-2 dataset, BERT fine-tuning and Data Split as example):
    ```bash
    python main.py --factor=data_split --mitigation_runs=100 --investigation_runs=10 --dataset=sst2 --experiment_type=finetuning --experiment_name=investigation -- configuration_name=stability --num_epochs=5 --model=bert --batch_size=8 --num_labelled=1000
    ```
1. The results from these runs will be saved to the folder specified by the `experiment_name`, `configuration_name`, `experiment_type`, `model`, `dataset` and `factor` arguments. The above command will save the results into the following path: `results/investigatigation_finetuning_bert_base/stability/sst2/data_split`. After the experiments are run, this folder should contain 100 folders `mitigation_{idx}` with idx ranging 0-99, each containing 10 folders `investigation_{idx}` with idx ranging 0-9.

To allow for reproducibility and unbiased comparison, we also provide arguments to set seeds that generate the configurations for the mitigation and investigation runs separately -- we provide default values for both.

To get the results from our paper, the investigation needs to be done for each factor and model in the following list (using the optimal hyperparameters and default investigation and mitigation seeds):
- In-context learning:
    - Factors: golden_model, data_split, label_choice, sample_choice, sample_order
    - Models: flan-t5, llama2
    - Experiment type: icl
- Fine-tuning:
    - Factors: golden_model, data_split, label_choice, sample_order, model_initialisation
    - Models: bert, roberta
    - Experiment type: finetuning
- Meta-learning:
    - Factors: golden_model, data_split, label_choice, sample_choice, sample_order, model_initialisation
    - Models: protonet, maml, reptile
    - Experiment type: meta_learning
- For all models the number of investigation runs should be set to 10 and the number of mitigation runs should be set to 100 (10 for LLaMA-2). For golden_model factor, the mitigation runs should be set to 1 000 (or 100 for LLaMA-2)  

Note: The golden model does not contain any `investigated randomness factor` -- therefore the number of investigation runs should be set to 1 and only the number of mitigation runs should be set to higher number


## Evaluating the experiments

For the evaluation purposes, we provide python script `process_results.py`. To run the script, the arguments at the top of the script need to be specified. In our example they should be set in following way:
- DATASET_PATH = 'sst2'
- MITIGATION_RUNS = 100
- INVESTIGATION_RUNS = 10
- MODEL = 'bert'
- L3D = 'finetuning'

We also provide arguments to only evaluate on a subsample of the results (arguments `TEST_DATA_FRACTION` specifying the fraction and `SEED` for controlling the random selection).

