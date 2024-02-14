# Spot the Bot: Inverse Problems of NLP

Repository for paper *Spot the Bot: Inverse Problems of NLP*, by Gromov V.A., Dang Q.N., Kogan A.S., Yerbolova A.S..

Used techniques:
- Clustering:

  algorithm type|crisp  |fuzzy        
  --------------|-------|-------------
  centroid-based|K-Means|C-Means      
  density-based |Wishart|Fuzzy Wishart
  
- Information theory
  
  Entropy-Complexity of ordinal patterns
  Generalised entropies
  
```
.
├── lib                      # includes full pipelines/
│   ├── trajectories         # text to semantic trajectory
│   ├── ordec                # implemented method for entropy-complexity calculations
│   ├── clustering           # clustering pipelines/
│   │   ├── pipelines
│   └── └── WishartFUZZY.py  # implemented Wishart algorithm for fuzzified data
└── examples                 # includes examples for implemented methods
```

## Classification experiments

Train dataset:
- 2000 literary texts
- 1000 YaLM-generated texts
- 1000 GPT-2-generated texts

Test dataset:
- 600 literary texts
- 300 LSTM-generated texts
- 300 mGPT-generated texts

Decision tree classifier:
|method |rus  |en   |ger  |vn   |fr   |
|-------|-----|-----|-----|-----|-----|
|ec     |0.769|0.824|0.971|0.948|0.642|
|wishart|0.562|0.705|0.710|0.646|0.635|
|fuzzy  |0.697|0.853|0.862|0.878|0.915|
|kmeans |0.973|0.860|0.633|0.666|0.691|
|fcmeans|0.930|0.782|0.675|0.730|0.635|
|all    |0.980|0.878|0.897|0.720|0.858|

Random forest classifier:
|method |rus  |en   |ger  |vn   |fr   |
|-------|-----|-----|-----|-----|-----|
|ec     |0.780|0.828|0.976|0.970|0.865|
|wishart|0.550|0.733|0.715|0.665|0.605|
|fuzzy  |0.695|0.854|0.891|0.813|0.926|
|kmeans |0.977|0.868|0.613|0.670|0.510|
|fcmeans|0.947|0.777|0.602|0.721|0.671|
|all    |0.992|0.908|0.912|0.717|0.857|

## Full pipeline

Black-box solution:
```bash
python lib/main.py --input_path="examples/The Picture of Dorian Gray.txt" --save_prediction_path=sample_predictions.json
```
