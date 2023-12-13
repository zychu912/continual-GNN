import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl

class GraphGenerator(nn.Module):
    def __init__(self, layers, node_count, condition_dim=10, batch_norm=True, temperature=1, threshold=0.02):
        super().__init__()
        self.node_count = node_count
        self.condition_dim = condition_dim
        self.temperature = temperature
        self.threshold = threshold
        self.fc_nodes = nn.Linear(layers[-1], layers[-1] * node_count)
        self.fc_adj = nn.Linear(layers[-1], node_count ** 2)
        self.fc_initial = nn.Linear(layers[0] + condition_dim, layers[1])
        
        modules = [self.fc_initial]
        for in_features, out_features in zip(layers[1:], layers[2:]):
            modules.append(nn.Linear(in_features, out_features))
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_features))
            modules.append(nn.ReLU())

        self.network = nn.Sequential(*modules)

    def forward(self, z, labels):
        num_labels = len(labels)
        nsample_per_label = z.size(0) // num_labels
        labels_list = [torch.full((nsample_per_label,), label, dtype=torch.long) for label in labels]
        label = torch.cat(labels_list)
        label = label[torch.randperm(z.size(0))]
        condition = F.one_hot(label, num_classes=self.condition_dim).to(z.device)
        x = torch.cat([z, condition], dim=1)
        x = self.network(x)
        feature_size = self.fc_nodes.out_features // self.node_count
        nodes = torch.sigmoid(self.fc_nodes(x)).view(-1, feature_size)
        adj_logits = self.fc_adj(x).view(-1, self.node_count, self.node_count)
        adj_probabilities = F.gumbel_softmax(adj_logits, tau=self.temperature, dim=-1)
        adj = (adj_probabilities > self.threshold).float()

        data_list = []
        for i in range(z.size(0)):
            node_features = nodes[i * self.node_count: (i + 1) * self.node_count, :]
            adj_matrix = adj[i]
            edge_index = adj_matrix.nonzero(as_tuple=False).t()
            graph_data = Data(x=node_features, edge_index=edge_index, y=label[i])
            data_list.append(graph_data)

        data_batch = Batch.from_data_list(data_list).to(z.device)
        return data_batch

class GraphDiscriminator(nn.Module):
    def __init__(self, layers, condition_dim=10, batch_norm=True):
        super().__init__()
        self.condition_dim = condition_dim
        self.conv1 = GCNConv(layers[0] + condition_dim, layers[1])
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(layers[1])
        modules = []
        for in_features, out_features in zip(layers[1:], layers[2:]):
            modules.append(GCNConv(in_features, out_features))
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_features))
            modules.append(nn.ReLU())
        self.network = nn.Sequential(*modules)
        self.fc_final = nn.Linear(layers[-1], 1)

    def forward(self, data):
        node_counts = torch.bincount(data.batch)
        expanded_labels = torch.repeat_interleave(data.y, node_counts, dim=0)
        condition = F.one_hot(expanded_labels, num_classes=self.condition_dim).to(data.x.device)
        x = torch.cat([data.x, condition], dim=1)
        x = F.relu(self.conv1(x, data.edge_index))
        for module in self.network:
            if isinstance(module, GCNConv):
                x = F.relu(module(x, data.edge_index))
            else:
                x = module(x)
        x = global_mean_pool(x, data.batch)
        return torch.sigmoid(self.fc_final(x))


class GraphGAN(pl.LightningModule):
    def __init__(self, gen_layers, disc_layers, node_count, batch_size, labels,
                 condition_dim=10, batch_norm=True, temperature=1, threshold=0.02, lr_gen=0.0005, lr_disc=0.0001, b1=0.5, b2=0.999):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.labels = labels
        self.generator = GraphGenerator(gen_layers, node_count, condition_dim, batch_norm, temperature, threshold)
        self.discriminator = GraphDiscriminator(disc_layers, condition_dim, batch_norm)
        self.automatic_optimization = False

    def forward(self, z, label):
        return self.generator(z, label)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        device = self.device
        z = torch.randn(self.batch_size, self.hparams.gen_layers[0]).to(device)
    
        opt_gen, opt_disc = self.optimizers()
        opt_gen.zero_grad()
        gen_data = self(z, self.labels)
        valid = torch.ones(self.batch_size, 1).to(device)
        g_loss = self.adversarial_loss(self.discriminator(gen_data), valid)
        self.manual_backward(g_loss)
        opt_gen.step()
        
        opt_disc.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(batch), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_data.detach()), torch.zeros(self.batch_size, 1).to(device))
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_disc.step()
    
        self.log('train_gen_loss', g_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('train_disc_loss', d_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        lr_gen = self.hparams.lr_gen
        lr_disc = self.hparams.lr_disc
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(b1, b2))
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc, betas=(b1, b2))
        return [opt_gen, opt_disc], []
