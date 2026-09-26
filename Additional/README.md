## Details of selected modules ##

| PTM         | lexical module | syntactic module | structural module |
|-------------|----------------|------------------|-------------------|
| BERT        | 1 - 2 layers   | 11–12 layers     | 7–8 layers        |
| CodeBERT    | 5–6 layers     | 1–2 layers       | 3–4 layers        |
| StarEncoder | 3–4 layers     | 8–9 layers       | 5–6 layers        |
| CodeGPT     | 8–9 layers     | 1–2 layers       | 4–5 layers        |
| PolyCoder   | 9–10 layers    | 3–4 layers       | 11–12 layers      |
| CodeGen     | 11–12 layers   | 1–2 layers       | 7–8 layers        |

## Details of GA settings and training strategy ##

The GA uses an initial random population of 8 chromosomes and evolves for 5 generations, involving 6 PTMs with 18 functional modules, with tournament selection (size 3), 2 elites, and a mutation probability of 0.3 for both module and connector mutations. The random seed is fixed at 42 to ensure reproducibility. 

After GA identifies the optimal module combination, the training process for the final checkpoint is the same as that of RQ1, i.e., trained for up to 100 epochs, and training stops after 10 consecutive epochs without validation F1 improvement, and the best checkpoint is used for testing, which ensures the adequacy of training and fairness in comparing performance and efficiency.
