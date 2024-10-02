# Deterministic investigation of the sensitivity of learning with limited labelled data to the effects of randomness

The repository contains the experiments and code for the following two papers:
- "On Sensitivity of Learning with Limited Labelled Data to the Effects of Randomness: Impact of Interactions and Systematic Choices" accepted at the EMNLP'24 main ([preprint](https://arxiv.org/abs/2402.12817)).
- "Comparing Specialised Small and General Large Language Models on Text Classification: 100 Labelled Samples to Achieve Break-Even Performance" as preprint on arXiv ([preprint](https://arxiv.org/abs/2402.12819)).

## Dependencies and local setup

The code in this repository uses Python. The required dependencies are specified in the `requirements.txt`. 

Simply run `pip install -r requiremets.txt`.

## Running the investigation

The investigation method considers multiple randomness factors (data split, label selection, model initialisation, data order, sample choice, model randomness, and a general golden model), multiple dataset from the GLUE and SuperGLUE benchmarks downloaded from HuggingFace and other multi-class models (SST2, CoLA, MRPC, BoolQ, AG News, TREC, SNIPS, DB Pedia) and multiple approaches and models for learning with limited labelled data (Fine-tuning with BERT and RoBERTa; Prompting and In-Context learning with Flan-T5, LLaMA-2, Mistral-7B, Zephyr-7B and ChatGPT; Instruction-Tuning with Flan-T5, Mistral-7B and Zephyr-7B; and Meta-Learning with MAML, FoMAML Reptile and Prototypical Networks). The investigation can be run with different set of parameters (check the `main.py` file for a set of accepted main and supplementary arguments for the investigation). 


We provide two separate sets of experiments that can be run using this repository: 
1. [Investigating the importance of individual randomness factor and the impact of interactions and systematic choices on this importance for different approaches for learning with limited labelled data and their sensitivity to the effects of randomness](EffectsOfRandomnessFactors.md)
1. [Investigating the impact of changing the number of labelled training samples on performance and stability/variability of results when comparing specialised small (fine-tuning, instruction-tuning) and general large language models (prompting, in-context learning)](ImpactOfDatasetSize.md)

Please refer to the detailed Readme files specific for the experiment of your interest (linked in the list above).

## Paper Citing

```
@inproceedings{pecher-etal-2024-sensitivity,
    title = "On Sensitivity of Learning with Limited Labelled Data to the Effects of Randomness: Impact of Interactions and Systematic Choices",
    author = "Pecher, Branislav  and
      Srba, Ivan  and
      Bielikova, Maria",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    year = "2024",
    publisher = "Association for Computational Linguistics",
}
```


