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

def main(job='classify', dataset=DGLdPLMGNNDataset, cv_taskname='S10998', et_taskname='S2814', collate_fn=collate_train_dgl_graphs,
         model_=GCNModel, params={'in_feats':1280, 'hidden_feats':256}, model_name='MLP'):
    result_dir = parent_dir + f'/pretrained/{cv_taskname}_{model_name}_{job}/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    train_dataset = dataset(cv_taskname, 'ESM-1v')
    batch_size = 128
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_models = [] 

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"Training fold {fold + 1}...")
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        model = model_(**params).to(device)
        if job=='classify':
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        num_epochs = 100
        patience = 8
        best_fold_val_loss = float('inf')
        counter = 0
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_preds.append(outputs)
                    val_labels.append(labels)
            val_loss /= len(val_loader)
            val_preds = torch.cat(val_preds).cpu().numpy()
            val_labels = torch.cat(val_labels).cpu().numpy()
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                counter = 0
                best_model = model
                torch.save(model.state_dict(),  f'{result_dir}/fold{fold + 1}_best_model.pth')
                np.savetxt(f'{result_dir}/fold{fold + 1}_val_preds.txt', val_preds)
                np.savetxt(f'{result_dir}/fold{fold + 1}_val_labels.txt', val_labels)
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered for fold", fold + 1)
                    break
        best_models.append(best_model)
        

    if et_taskname is not None:
        print("Evaluating on the test set...")
        test_dataset = dataset(et_taskname, 'ESM-1v')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_preds = []
        for fold, model in enumerate(best_models):
            model.eval()
            fold_preds = []
            fold_labels = []
            # model.load_state_dict(torch.load(result_dir + f'/fold{fold + 1}_best_model.pth'))
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


if __name__ == "__main__":
    main(job='classify', dataset=DGLdPLMGNNDataset, cv_taskname='S10998', et_taskname='S2814', model_=GCNModel, params={'in_feats':1280, 'hidden_feats':256}, model_name='GCN')
    main(job='classify', dataset=DGLdPLMGNNDataset, cv_taskname='S10998', et_taskname='S2814', model_=GATModel, params={'in_feats':1280, 'hidden_feats':256}, model_name='GAT')
