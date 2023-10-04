# Investigating the impact of changing the number of labelled training samples on performance and stability/variability of results in fine-tuning, prompting, in-context learning and instruction-tuning approaches

## Dependencies and local setup

The code in this repository uses Python. The required dependencies are specified in the `requirements.txt`. 

Simply run `pip install -r requiremets.txt`.

## Running the investigation

To run the specific experiment, follow these steps:

1. Install the requirements.
1. Choose dataset to run the investigation on. Currently we support following options: "sst2", "mrpc", "cola", "boolq", "rte". However, as we use the HuggingFace, the set of datasets can be easily extended to include other ones (the dataset classes in `data.py` file needs to be extended with the loading and processing of the new dataset). 
1. Choose the training dataset size to run the investigation on.
1. Choose number of runs for the investigation.
1. Run the investigation using following command (with SST-2 dataset, LLaMA-2 in-context learning on a subset of 1 000 training and test samples):
    ```bash
    python main.py --factor=golden_model --mitigation_runs=100 --investigation_runs=1 --dataset=sst2 --experiment_name=dataset_size_change --experiment_type=icl --model=llama2 --configuration_name=num_samples_1000 --num_labelled=1000 --num_labelled_test=1000 --full_test=0 
    ```
1. The results from these runs will be saved to the folder specified by the `experiment_name`, `configuration_name`, `experiment_type`, `model`, `dataset` and `factor` arguments. The above command will save the results into the following path: `results/dataset_size_change_icl_llama2_base/num_samples_1000/sst2/golden_model`. After the experiments are run, this folder should contain 100 folders `mitigation_{idx}` with idx ranging 0-99, each containing 1 folder `investigation_0` with the results.

To allow for reproducibility and unbiased comparison, we also provide arguments to set seeds that generate the configurations for the mitigation and investigation runs separately -- we provide default values for both.

To get the results from our paper, the investigation needs to be done for each model and dataset in the following list (using the optimal hyperparameters, the default mitigation and investigation seeds, and settings described in the paper):
- Datasets: sst2, mrpc, boolq
- Models:
    - (experiment_type) finetuning: bert, roberta
    - (experiment_type) prompting: flan-t5, llama2, chatgpt
    - (experiment_type) icl: flan-t5, llama2, chatgpt
    - (experiment_type) instruction_tuning_steps: flan-t5
- The number of mitigations runs for all models should be set to 100 (except for chatgpt where it should be just 6).

## Evaluating the experiments

For the evaluation purposes, we provide python script `process_dataset_size_change_results.py`. For the script to run correctly, make sure all the parameters are set according to the experiments (dataset sizes)

We also provide additional python script `process_dataset_size_change_results_threshold.py`, which can be used for more detailed analysis of the cross-over points.