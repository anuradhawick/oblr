import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Dataset
from torch_geometric.data import NeighborSampler
import numpy as np
import argparse
import os


class SAGE(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, device):
        super(SAGE, self).__init__()

        self.num_layers = num_layers
        hidden_channels = (in_channels + out_channels)//2

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

        self.device = device
        self.to(device)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        x = self.fc1(x)
        embedding = x
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.fc2(x)

        return x.log_softmax(dim=-1), embedding

    def inference(self, x_all, subgraph_loader):
        idx = []

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                if i == 0:
                    idx += list(n_id[:batch_size].numpy())
                edge_index, _, size = adj.to(self.device)
                x = x_all[n_id].to(self.device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = F.relu(x)
                xs.append(x)

            x_all = torch.cat(xs, dim=0)

        x = self.fc1(x_all)
        x = F.relu(x)
        x = self.fc2(x)

        x = x.cpu()

        return np.array(idx), x


def train(model, x, y, optimizer, train_loader, device):
    model.train()
    total_loss = 0

    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out, embd = model(x[n_id], adjs)

        loss = F.nll_loss(out, y[n_id[:batch_size]])

        loss.backward()
        optimizer.step()

        total_loss += float(loss)

    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def predict_all(model, x, subgraph_loader):
    model.eval()
    out = model.inference(x, subgraph_loader)

    return out


def get_graph_data(features, edges):
    edge_index = torch.tensor(edges, dtype=torch.long)
    data = Data(x=torch.tensor(features).float(),
                edge_index=edge_index.t().contiguous())

    return data


def get_train_data(features, read_cluster):
    train_idx = read_cluster.T[0]
    train_idx = torch.LongTensor(train_idx)

    y = -1 * torch.ones(len(features), dtype=torch.long)
    y[train_idx] = torch.LongTensor(read_cluster.T[1])
    no_classes = len(set(read_cluster.T[1]))

    return train_idx, y, no_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""OBLR GraphSAGE Routine.""")

    parser.add_argument('--data', '-d',
                        help="Data from build_graph.py data.npz",
                        type=str,
                        required=True)
    parser.add_argument('--ground-truth', '-g',
                        help="Ground truth of reads for dry runs and sensitivity tuning",
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument(
        '--output', '-o', help="Output directory", type=str, required=True)

    args = parser.parse_args()

    data_path = args.data
    has_truth = args.ground_truth is not None
    output = args.output

    if not os.path.isdir(output):
        os.mkdir(output)

    # essential data
    data = np.load(data_path)
    edges = data['edges']
    comp = data['scaled']
    read_cluster = data['read_cluster']

    if has_truth:
        truth = np.array(open(args.grount_truth).read().strip().split("\n"))

    # create dataset
    data = get_graph_data(comp, edges)
    train_idx, y, no_classes = get_train_data(comp, read_cluster)

    # sampler for training
    train_loader = NeighborSampler(data.edge_index,
                                   node_idx=train_idx,
                                   sizes=[50, 50],
                                   batch_size=64,
                                   pin_memory=True,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=8)
    # sampler for predicting
    subgraph_loader = NeighborSampler(data.edge_index,
                                      node_idx=None,
                                      sizes=[100],
                                      pin_memory=True,
                                      batch_size=10240,
                                      shuffle=False,
                                      num_workers=8)

    # optimizer and running device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAGE(data.x.shape[1], no_classes, 2, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=10e-6)

    print(f"Using device: {device}")
    print(f"Sage Model \n {model}")

    x = data.x.to(device)
    y = y.to(device)

    # actual training
    epochs = 100
    losses = []
    prev_loss = 100

    for epoch in range(1, epochs+1):
        loss = train(model, x, y, optimizer, train_loader, device)
        dloss = prev_loss-loss
        prev_loss = loss
        losses.append(loss)

        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

        if loss < 0.05:
            print('Early stopping, loss less than 0.05')
            break

    # final result
    idx, preds = predict_all(model, x, subgraph_loader)
    classes = torch.argmax(preds, axis=1)
    np.savez(output + '/classes.npz', classes=classes.numpy(), classified=idx)
