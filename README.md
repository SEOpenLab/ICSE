# ICSE-2027
This is data and code for ProbeFuse submited to ICSE-2027



## Instructions
This is the experimental process code and data of the proposed method Coding-Fuse and FMF method, which can be found in the code directory.

If you need to repeat the performance values ​​of the experimental results, we recommend running the test code directly, in each setcode parameter --test part. But before that, you need to (1) first copy the dataset folder to the directory at the same level as code in each fusion folder. (2) Secondly, import the code pre-trained models in the code folder, and we have prepared the file name for you, prepared code PTMs can be found at https://zenodo.org/records/15543665 (3) Finally, you need to import our trained test model (can be found at https://zenodo.org/records/15543870, https://zenodo.org/records/15544997 and https://zenodo.org/records/15546108) in saved_models in the code folder. You can get the same performance values ​​as us.

If you need to perform the entire experiment, that is, the entire process from training to testing, we recommend that you directly execute all the commands of setcode, but before that you also need to perform the first two steps above.

## Preliminary Study
### Probe Datasets and Tasks
We employ the same experimental procedure as Karmakar et al. \cite{DBLP:conf/kbse/KarmakarR21}, using 1000 Java samples with a uniform class distribution, i.e., split ratio of 6:2:2, from the JEMMA dataset on the three tasks, i.e., Identifier Tagging (IDN), AST Node Tagging (AST), Cyclomatic Complexity (CPX) for probing code attribute information of code PTMs layer by layer. 
### Probe Code and Execute
First we will create and activate a virtual environment with anaconda, the Probe_GA.tar.gz is our environment, you could put it in your own computer. 

$conda activate Probe_GA

Then we begin probing the code pre-trained models layer by layer using probe datasets and tasks, and we will get the scores of each layer for three code attribute of code PTMs.

$python probe_extractor.py & python probe_classifier-accuracy.py > results/results.txt

### Sliding Window Strategy
We use a sliding window strategy to calculate the sum of the accuracy of consecutive layers of length 2, and return the result with the highest score, which means each module consists of two consecutive layers with the highest accuracy. Then, each code PTM is divided into three modules: the two consecutive layers with the richest lexical information constitute the lexical module, the two consecutive layers with the richest syntactic information constitute the syntactic module, and the two consecutive layers with the richest structural information constitute the structural module and semantic knowledge is implicitly preserved within all selected modules.

We input the results obtained above into the INPUT PROBE RESULTS section of the modular.py file, and executing the file will produce the Top-3 candidate modules selected according to a two-layer sliding strategy.

$python modular.py

Then we get 18 code functional modules, each code PTM have three code modules.


### RQ1


### RQ2



## GA Search 



## Details
### Dataset Partition
code clone detection: Train-901028, Valid-415416, Test-415416

technical debt detection: Train-23012, Valid-7674, Test-7674




### Finetune 

For code clone detection, use 10% Train and 10% Valid for fine-tuning, for code smell detection and technical debt detection, use full Train and Valid for fine-tuning.

### Hyperparameters
The training and testing commands and hyperparameters can be found in the setcode files in each folder. Training hyperparameters such as learning rate, batch size, and random seed are aligned with FMF and OSM in the same task and scenario to ensure fair and optimal comparisons.


If you have any questions, you can leave a message.

