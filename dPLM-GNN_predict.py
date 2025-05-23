import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv
from torch.utils.data import DataLoader
import torch.optim as optim
parent_dir = os.path.abspath(os.path.dirname(__file__))
from dgl.dataloading import GraphDataLoader
from process_dataset import build_dPLM_DGL_graphs,generate_esm_embeddings_local

def collate_train_dgl_graphs(sample):
    graphs, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)

class DGLdPLMGNNDataset(Dataset):
    def __init__(self, task_name, plm_name='ESM-1V'):
        task_dir = f'task/{task_name}'
        graph_dir = f"{task_dir}/{plm_name}_DGL_graphs"
        df = pd.read_csv(f"{task_dir}/{task_name}_processed.csv", sep='\t')
        self.graph_files = [os.path.join(graph_dir, f"{row['UniprotID']}-{row['Mutation']}.bin") for index,row in df.iterrows()]
        self.labels = df['Label'].values  

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph = dgl.load_graphs(self.graph_files[idx])[0][0]
        graph = dgl.add_self_loop(graph)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return graph, label


class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, job="classify"):
        super(GCNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, hidden_feats, allow_zero_in_degree=True)
        if job == "classify":
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(hidden_feats, 1),
                torch.nn.Sigmoid()
            )
        else:
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(hidden_feats, 1),
            )    

    def forward(self, graph):
        h = graph.ndata['n'] 
        h = F.relu(self.conv1(graph, h))  
        h = self.conv2(graph, h)
        graph.ndata['h'] = h
        graph_embedding = dgl.mean_nodes(graph, 'h') 
        probs = self.readout(graph_embedding) 
        return probs.squeeze()

class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads=4, job="classify"):
        super(GATModel, self).__init__()
        self.num_heads = num_heads
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads, allow_zero_in_degree=True)
        self.gat2 = GATConv(hidden_feats * num_heads, hidden_feats, 1, allow_zero_in_degree=True)
        if job == "classify":
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(hidden_feats, 1),
                torch.nn.Sigmoid()
            )
        else:
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(hidden_feats, 1),
            )    

    def forward(self, graph):
        h = graph.ndata['n'] 
        h = self.gat1(graph, h)
        h = F.elu(h)
        h = h.flatten(1)
        h = self.gat2(graph, h)
        h = h.squeeze(1)
        graph.ndata['h'] = h
        graph_embedding = dgl.mean_nodes(graph, 'h')
        probs = self.readout(graph_embedding)
        return probs.squeeze(-1)

def predict(job='classify', dataset=DGLdPLMGNNDataset, cv_taskname='S10998', et_taskname='S2814', collate_fn=collate_train_dgl_graphs, 
         model_=GCNModel, params={'in_feats':1280, 'hidden_feats':256}, model_name='MLP'):
    result_dir = parent_dir + f'/pretrained/{cv_taskname}_{model_name}_{job}/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    print("Evaluating on the test set...")
    test_dataset = dataset(et_taskname, 'ESM-1v')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_preds = []
    for fold in range(5):
        fold_preds = []
        fold_labels = []
        model = model_(**params).to(device)
        model.load_state_dict(torch.load(result_dir + f'/fold{fold + 1}_best_model.pth'))
        model.eval()
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                fold_preds.append(outputs)
                fold_labels.append(labels)
        fold_preds = torch.cat(fold_preds).cpu().numpy()
        fold_labels = torch.cat(fold_labels).cpu().numpy()
        test_preds.append(fold_preds)
        if fold == 0:
            test_labels = np.array(fold_labels)
    test_preds = np.mean(np.array(test_preds), axis=0)
    np.savetxt(result_dir + f'/test_{et_taskname}_preds.txt', test_preds)
    np.savetxt(result_dir + f'/test_{et_taskname}_labels.txt', test_labels)


def process_predict(task_name):
    generate_esm_embeddings_local(task_name=task_name, protein_type='WT', single=True)
    generate_esm_embeddings_local(task_name=task_name, protein_type='Mut', single=True)
    build_dPLM_DGL_graphs(task_name, 'ESM-1v')
    predict(job='classify', dataset=DGLdPLMGNNDataset, cv_taskname='S10998', et_taskname=task_name, model_=GATModel, params={'in_feats':1280, 'hidden_feats':256}, model_name='GAT')



if __name__ == "__main__":
    process_predict('G4U3G8')
    
