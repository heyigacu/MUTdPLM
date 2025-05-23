import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch.utils.data import Dataset, Subset

parent_dir = os.path.abspath(os.path.dirname(__file__))


class dPLMMLPDataset(Dataset):
    def __init__(self, task_name, plm_name='ESM-1V'):
        task_dir = f'task/{task_name}'
        df = pd.read_csv(f"{task_dir}/{task_name}_processed.csv", sep='\t')
        self.features = np.load(f"{task_dir}/{plm_name}_WT_seq_embed.npz")['data'] - \
                        np.load(f"{task_dir}/{plm_name}_Mut_seq_embed.npz")['data']
        self.labels = df['Label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def main(cv_taskname='S10998', et_taskname='S2814', model_name='RFC'):
    result_dir = parent_dir + f'/pretrained/{cv_taskname}_{model_name}_classify/'
    os.makedirs(result_dir, exist_ok=True)

    train_dataset = dPLMMLPDataset(cv_taskname, 'ESM-1v')
    X = train_dataset.features
    y = train_dataset.labels

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold + 1}...")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = RandomForestClassifier(n_estimators=100, random_state=fold)
        model.fit(X_train, y_train)

        val_preds = model.predict_proba(X_val)[:, 1]
        val_labels = y_val

        np.savetxt(f'{result_dir}/fold{fold + 1}_val_preds.txt', val_preds)
        np.savetxt(f'{result_dir}/fold{fold + 1}_val_labels.txt', val_labels)

        models.append(model)

    if et_taskname is not None:
        print("Evaluating on test set...")
        test_dataset = dPLMMLPDataset(et_taskname, 'ESM-1v')
        X_test = test_dataset.features
        y_test = test_dataset.labels
        test_preds_list = []

        for fold, model in enumerate(models):
            preds = model.predict_proba(X_test)[:, 1]
            test_preds_list.append(preds)

        test_preds = np.mean(np.array(test_preds_list), axis=0)
        np.savetxt(f'{result_dir}/test_{et_taskname}_preds.txt', test_preds)
        np.savetxt(f'{result_dir}/test_{et_taskname}_labels.txt', y_test)


if __name__ == "__main__":
    main(cv_taskname='S10998', et_taskname='S2814')
    main(cv_taskname='M576', et_taskname='M167')
    main(cv_taskname='ProteinGym_clinical', et_taskname=None)
