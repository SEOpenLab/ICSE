# ICSE-2027
This is data and code for ProbeFuse submited to ICSE-2027



## Instructions
This is the experimental process code and data of the proposed method ProbeFuse.

If you want to check the result obtained in the paper, we also provide the data, code, results, models and running log in total of 88GB zip file of our experiments, which is the same folder name in the *Zenodo* as this repository (*[10.5281/zenodo.21072012](https://doi.org/10.5281/zenodo.21072012)*). 

If you need to repeat the performance values ​​of the experimental results and further design, we recommend running the code according to the next guidance.

## Preliminary Study
### Probe Datasets and Tasks
We employ the same experimental procedure as Karmakar et al. \cite{DBLP:conf/kbse/KarmakarR21}, using 1000 Java samples with a uniform class distribution, i.e., split ratio of 6:2:2, from the JEMMA dataset on the three tasks, i.e., Identifier Tagging (IDN), AST Node Tagging (AST), Cyclomatic Complexity (CPX) for probing code attribute information of code PTMs layer by layer. 
### Probe Code and Execute
First we will create and activate a virtual environment with anaconda, the Probe_GA.tar.gz is our environment, you could put it in your own computer. 
   ```   
   $ conda activate Probe_GA
   ```   
Then we begin probing the code pre-trained models layer by layer using probe datasets and tasks, and we will get the scores of each layer for three code attribute of code PTMs.
   ```   
   $ python probe_extractor.py & python probe_classifier-accuracy.py > results/results.txt
   ```   

### Sliding Window Strategy
We use a sliding window strategy to calculate the sum of the accuracy of consecutive layers of length 2, and return the result with the highest score, which means each module consists of two consecutive layers with the highest accuracy. Then, each code PTM is divided into three modules: the two consecutive layers with the richest lexical information constitute the lexical module, the two consecutive layers with the richest syntactic information constitute the syntactic module, and the two consecutive layers with the richest structural information constitute the structural module and semantic knowledge is implicitly preserved within all selected modules.

We input the results obtained above into the INPUT PROBE RESULTS section of the modular.py file, and executing the file will produce the Top-3 candidate modules selected according to a two-layer sliding strategy.
   ```   
   $ python modular.py
   ```   
Then we get 18 code functional modules, each code PTM have three code modules.

### RQ1
For RQ1, in the RQ1 folder, there are two task folders, and each task has two subfolders: one for the original model training and fine-tuning, and the other for the training and fine-tuning in modular fusion. These modules come from three different functional modules of the same model. Simply execute the instructions in the setcode file to run the experiment. For example, 

   ```  
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=./codebert-base \
    --model_name_or_path=./codebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 100 \
    --early_stopping_patience 10 \
    --block_size 50 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log && python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=./codebert-base \
    --model_name_or_path=./codebert-base \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 100 \
    --early_stopping_patience 10 \
    --block_size 50 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test.log && python ../evaluator/evaluator.py -a ../dataset/test.jsonl -p saved_models/predictions.txt
   ```
It is worth noting that in order to ensure the fairness of the experiment, we used the same training strategy, random seed, training and testing batches, and other parameters, which were not mentioned in the paper.

### RQ2
For RQ2, in the RQ2 folder, there are also two task folders, and each task has 16 subfolders. Each folder represents a module combination. Due to the large number of combinations, we only conducted experiments on a small subset of functional module combinations to verify the feasibility of the scheme. For each folder, executing the instructions in the setcode file will run the training and testing strategy for each combination. For example, 

   ```  
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=./codebert-base \
    --model_name_or_path=./codebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 100 \
    --early_stopping_patience 10 \
    --block_size 50 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log && python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=./codebert-base \
    --model_name_or_path=./codebert-base \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 100 \
    --early_stopping_patience 10 \
    --block_size 50 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test.log && python ../evaluator/evaluator.py -a ../dataset/test.jsonl -p saved_models/predictions.txt
   ```
Note that the parameter settings here are the same as in RQ1, and the parameter settings are also the same for different combinations. This controls these variables to evaluate only the effectiveness of the functional module combination scheme.

## GA Search-RQ3+RQ4
The GA code is located in the RQ3+RQ4 folder, which contains two sub-files corresponding to the GA code for the two tasks. The parameters during the GA search process are defined in the code, and the training and testing parameters after finding the optimal module combination are the same as those for RQ1 and RQ2 to ensure a fair performance comparison.

To perform the best functional module search and subsequent training/testing for code clone detection task, execute the commands in the `setcode` folder within this directory. For example,

   ```
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=./codebert-base \
    --model_name_or_path=./codebert-base \
    --tokenizer_name=roberta-base \
    --do_GA \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --epoch 100 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee GA.log && python ../evaluator/evaluator.py -a ../dataset/test.txt -p saved_models/predictions.txt
   ```


### Finetune details
For code clone detection, use 10% Train and 10% Valid for fine-tuning, for code smell detection and technical debt detection, use full Train and Valid for fine-tuning.

If you have any questions, you can leave a message.

