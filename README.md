# Deterministic investigation of stability/result variability due to randomness in learning with limited labelled data on text data

## Dependencies and local setup

The code in this repository uses Python. The required dependencies are specified in the `requirements.txt`. 

Simply run `pip install -r requiremets.txt`.

## Running the investigation

The investigation method considers multiple randomness factors (data split, label selection, model initialisation, data order, sample choice, model randomness, and a general golden model), multiple dataset from the GLUE and SuperGLUE benchmarks downloaded from HuggingFace and other multi-class models (SST2, CoLA, MRPC, BoolQ, AG News, TREC, SNIPS, DB Pedia) and multiple approaches and models for learning with limited labelled data (Fine-tuning with BERT and RoBERTa; Prompting and In-Context learning with Flan-T5, LLaMA-2, Mistral-7B, Zephyr-7B and ChatGPT; Instruction-Tuning with Flan-T5; and Meta-Learning with MAML, FoMAML Reptile and Prototypical Networks). We provide an optimal set of parameters for each model and each dataset that were used in our experiments in `optimals_parameters` folder. However, the investigation can be run with different set of parameters (check the `main.py` file for a set of accepted main and supplementary arguments for the investigation). 


We provide two separate sets of experiments that can be run using this repository: 
1. [Investigating effects of randomness factor on the stability of the results, while taking interactions between randomness factors into consideration](EffectsOfRandomnessFactors.md)
1. [Investigating the impact of changing the number of labelled training samples on performance and stability/variability of results in fine-tuning, prompting, in-context learning and instruction-tuning approaches](ImpactOfDatasetSize.md)

Please refer to the detailed Readme files specific for the experiment of your interest (linked in the list above).



