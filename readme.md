## install dependencies (for example cuda12.1)
torch_geometric, torch, dgl, fair-esm
```
pip3 install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -c dglteam/label/cu121 dgl
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch_geometric
pip install fair-esm
```

## run according to the paper

```
python process_dataset.py
python dPLM-GNN.py
python dPLM-MLP.py
python dPLM-RFC.py
python dPLM-SVC.py
python metric.py
```


## predict
If you want to perform your own prediction, here we take G4U3G8 as an example. Please create a folder named G4U3G8 under the task directory, and then create a file named G4U3G8_processed.csv under task/G4U3G8/. This file should contain five columns: UniprotID, WTSequence, MutSequence, Mutation, and Label.
(UniprotID will be used to find the structure file {UniprotID}.pdb in the pdbs/ folder. WTSequence and MutSequence are the wild-type and mutated sequences, Mutation is the mutation name and should be as unique as possible, and Label is the label which can be random.)

Also, place your predicted structure file (e.g., G4U3G8.pdb) in the pdbs directory, and ensure the filename matches the UniprotID.
And then run:
```
python dPLM-GNN_predict.py
```