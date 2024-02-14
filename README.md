# Spot the Bot: Semantic Analysis of Natural Language Paths

Used techniques:
- Clustering:

  algorithm type|crisp  |fuzzy        
  --------------|-------|-------------
  centroid-based|K-Means|C-Means      
  density-based |Wishart|Fuzzy Wishart
  
- Information theory
  
  Entropy-Complexity of ordinal patterns
- Topological data analysis

  Vietoris-Rips filtration + H0- and H1-diagram features

```
.
├── lib                      # includes full pipelines/
│   ├── trajectories         # text to semantic trajectory
│   ├── ordec                # implemented method for entropy-complexity calculations
│   ├── clustering           # clustering pipelines/
│   │   ├── pipelines
│   │   └── WishartFUZZY.py  # implemented Wishart algorithm for fuzzified data
│   └── tda                  # features using tda                   
├── examples                 # includes examples for implemented methods
└── results                  # includes resulting tables
```

## Full pipeline

Black-box solution:
TODO: add fuzzy wishart features, model retraining, hyperparameter selection.

```bash
python lib/main.py --input_path="examples/The Picture of Dorian Gray.txt" --save_prediction_path=sample_predictions.json
```

Pipeline parameters:

- `input_path`: path to text file
- `lang`: `english/russian`, language of the text
- `split_text`: if set, split text into paragraphs and predict per paragraph

- `wdict_path`: path to word dictionary with .npy extension
- `wdim`: word embedding dimension (8 by default)
- `n`: number of words in ngram (2 by default)
- `method`: clustering method (kmeans, fcmeans, wishart) or entropy-complexity method ('ec')
- `k`: number of clusters in kmeans/fcmeans; number of neighbours in Wishart
- `clf_path`: path to pretrained sklearn classifier model, .pkl extension
- `save_prediction_path`: path to save results as json file


## Final classification

Train dataset:
- 2000 literary texts
- 1000 balaboba-generated texts
- 1000 gpt2-generated texts

Test dataset:
- 600 literary texts
- 300 lstm-generated texts
- 300 mgpt-generated texts

[Decision tree classifier](res_clf_decision_tree.csv):
|method |rus  |en   |ger  |vn   |fr   |
|-------|-----|-----|-----|-----|-----|
|ec     |0.769|0.824|0.971|0.948|0.642|
|wishart|0.562|0.705|0.710|0.646|0.635|
|fuzzy  |0.697|0.853|0.862|0.878|0.915|
|kmeans |0.973|0.860|0.633|0.666|0.691|
|fcmeans|0.930|0.782|0.675|0.730|0.635|
|all    |0.980|0.878|0.897|0.720|0.858|

[Random forest classifier](res_clf_random_forest.csv)
|method |rus  |en   |ger  |vn   |fr   |
|-------|-----|-----|-----|-----|-----|
|ec     |0.780|0.828|0.976|0.970|0.865|
|wishart|0.550|0.733|0.715|0.665|0.605|
|fuzzy  |0.695|0.854|0.891|0.813|0.926|
|kmeans |0.977|0.868|0.613|0.670|0.510|
|fcmeans|0.947|0.777|0.602|0.721|0.671|
|all    |0.992|0.908|0.912|0.717|0.857|
