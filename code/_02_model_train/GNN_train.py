from code._01_preprocessing.GNN_preproc import GNN_smiles_to_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,GINEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, average_precision_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, average_precision_score
import os
import joblib

def GNN_transform_data(df):
    """ Divise les données en ensembles d'entraînement et de validation. """
    X=df[['molecule_smiles']]
    y=df['binds']
    X['graph_data']=X['molecule_smiles'].apply(GNN_smiles_to_graph)
    X_train,X_test,y_train,y_test = train_test_split(X['graph_data'],y,test_size=0.3,random_state=42)


    train_dataset = [Data(x=graph.x, edge_index=graph.edge_index,edge_attr=graph.edge_attr, y=torch.tensor([label], dtype=torch.float))
                    for graph, label in zip(X_train, y_train)] # liste de couple de Datas avec graph + y
    test_dataset = [Data(x=graph.x, edge_index=graph.edge_index,edge_attr=graph.edge_attr, y=torch.tensor([label], dtype=torch.float))
                    for graph, label in zip(X_test, y_test)]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    num_node_features=X['graph_data'].iloc[0].x.shape[1]
    num_edge_features=X['graph_data'].iloc[0].edge_attr.shape[1]
    return(X_train,X_test,y_train,y_test,train_loader,test_loader,num_edge_features,num_node_features)


def GNN_create_classes():
    class GNN(torch.nn.Module):
        def __init__(self, num_node_features, num_edge_features, hidden_channels, num_layers, dropout_rate):
            super(GNN, self).__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(GINEConv(torch.nn.Linear(num_node_features, hidden_channels), edge_dim=num_edge_features))
            for _ in range(num_layers - 1):
                self.convs.append(GINEConv(torch.nn.Linear(hidden_channels, hidden_channels), edge_dim=num_edge_features))
            self.fc1 = torch.nn.Linear(hidden_channels, 64)
            self.fc2 = torch.nn.Linear(64, 1)
            self.dropout = torch.nn.Dropout(p=dropout_rate)

            # Initialize weights
            self.apply(self.init_weights)

        def init_weights(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        def forward(self, data):
            x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
            for conv in self.convs:
                x = conv(x, edge_index, edge_attr)
                x = F.relu(x)
                x = self.dropout(x)
            x = global_mean_pool(x, batch)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    class GNNModel(BaseEstimator, ClassifierMixin):
        def __init__(self, num_node_features, num_edge_features, hidden_channels=128, num_layers=2, dropout_rate=0.0, learning_rate=0.01, batch_size=64, epochs=50, weight_decay=0.0):
            self.num_node_features = num_node_features
            self.num_edge_features = num_edge_features
            self.hidden_channels = hidden_channels
            self.num_layers = num_layers
            self.dropout_rate = dropout_rate
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs
            self.weight_decay = weight_decay
            self.model = GNN(num_node_features, num_edge_features, hidden_channels, num_layers, dropout_rate)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.criterion = torch.nn.BCEWithLogitsLoss()

        def fit(self, X_train, y_train):
            self.classes_ = np.unique(y_train)  # Ajout de classes_ ici
            self.model.train()
            train_dataset = [Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=torch.tensor([label], dtype=torch.float))
                            for graph, label in zip(X_train, y_train)]
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.epochs):
                total_loss = 0
                for batch in train_loader:
                    self.optimizer.zero_grad()
                    out = self.model(batch)
                    batch.y = batch.y.view(-1, 1)
                    loss = self.criterion(out, batch.y)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            return self

        def predict_proba(self, X):
            self.model.eval()
            test_dataset = [Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr) for graph in X]
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            all_outputs = []
            with torch.no_grad():
                for batch in test_loader:
                    out = self.model(batch)
                    prob_class_1 = torch.sigmoid(out).detach().cpu().numpy()
                    prob_class_0 = 1 - prob_class_1
                    probs = np.hstack((prob_class_0, prob_class_1))  # Concaténer les probabilités pour les deux classes
                    if np.isnan(probs).any():
                        raise ValueError("NaN detected in probabilities")
                    all_outputs.extend(probs)
            return np.array(all_outputs)


        def predict(self, X):
            proba = self.predict_proba(X)
            return (proba >= 0.5).astype(int)  # Convert to binary predictions

    return GNN,GNNModel

def GNN_find_best_params(GNNModel,X_train,X_test,y_train,y_test,num_node_features,num_edge_features):
    param_grid = {
        'hidden_channels': [64,128, 256],  # Réduire à deux valeurs importantes
        'learning_rate': [0.01,0.05, 0.001],  # Explorer deux valeurs typiques
        'num_layers': [2, 3],  # Réduire à deux options
    }

    # Scorer pour l'average precision
    scorer = make_scorer(average_precision_score, response_method='predict_proba')


    # GridSearchCV avec les paramètres réduits
        # GridSearchCV avec les paramètres réduits
    grid_search = GridSearchCV(GNNModel(num_node_features=num_node_features, num_edge_features=num_edge_features),
                            param_grid,
                            scoring=scorer,
                            cv=2,  # 3-fold cross-validation
                            verbose=3)

    # Exécuter la recherche en grille
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres et le score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    # Évaluer sur l'ensemble de test
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred_proba = best_model.predict_proba(X_test)

    # Sélectionner uniquement les probabilités de la classe positive
    y_pred_proba_class_1 = y_pred_proba[:, 1]

    # Calculer la précision moyenne pour la classe positive
    test_score = average_precision_score(y_test, y_pred_proba_class_1)
    print("Test set average precision score: ", test_score)

    return best_params

def GNN_find_best_model(GNN,best_params,num_node_features,num_edge_features):
    model = GNN(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_channels=best_params['hidden_channels'],
            num_layers=best_params['num_layers'],
            dropout_rate=0.0  # Supposons que le dropout_rate soit constant
        )
    return model

def GNN_save_model(model, name_protein, nb_sample):
    """ Enregistre le meilleur modèle trouvé. """
    parent_dir = os.path.dirname(os.getcwd())
    model_path = os.path.join(parent_dir, f'drug_smile/raw_data/best_model_vector_SVC_{name_protein}_{nb_sample}.pkl\n')
    joblib.dump(model, model_path)
    print(f"\nModèle enregistré à : {model_path}")



def GNN_train(best_model,best_params,train_loader,test_loader):
    # Utiliser les meilleurs hyperparamètres trouvés pour réentraîner le modèle si nécessaire
    patience = 15
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    criterion = torch.nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    average_precisions = []
    patience_counter = 0

    best_val_loss= float('inf')

    for epoch in range(10000):
        best_model.train()
        all_targets = []
        all_outputs = []
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            out = best_model(batch)
            batch.y = batch.y.view(-1, 1)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_outputs.append(torch.sigmoid(out).detach().cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        best_model.eval()
        val_loss = 0

        with torch.no_grad():
            all_val_targets = []
            all_val_outputs = []
            for batch in test_loader:  # val_loader pour le set de validation
                out = best_model(batch)
                batch.y = batch.y.view(-1, 1)
                loss = criterion(out, batch.y)
                val_loss += loss.item()

                all_val_outputs.append(torch.sigmoid(out).detach().cpu().numpy())
                all_val_targets.append(batch.y.cpu().numpy())

            avg_val_precision = average_precision_score(np.concatenate(all_val_targets), np.concatenate(all_val_outputs))

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        average_precisions.append(avg_val_precision)
        if best_val_loss<avg_val_loss :
            patience_counter +=1
        else :
            best_val_loss=avg_val_loss
            patience_counter=0
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Avg Precision: {avg_val_precision:.4f}')
