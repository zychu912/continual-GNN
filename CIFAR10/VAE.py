import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import TensorDataset

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, n_labels, label_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(n_labels, label_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + label_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        labels_embed = self.label_embedding(labels)
        x_combined = torch.cat([x, labels_embed], dim=1)
        encoded = self.encoder(x_combined)
        mu, logvar = torch.chunk(encoded, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        z_combined = torch.cat([z, labels_embed], dim=1)
        return self.decoder(z_combined), mu, logvar

class LitCVAE(pl.LightningModule):
    def __init__(self, gcn, input_dim, latent_dim, label_dim, lr, n_labels=10, batch_size=64):
        super().__init__()
        self.gcn = gcn
        self.vae = CVAE(input_dim, latent_dim, n_labels, label_dim)
        self.batch_size = batch_size
        self.lr = lr

    def forward(self, data):
        emb, y = self.gcn.get_emb(data)
        return self.vae(emb, y)

    def compute_loss(self, emb, y, x_hat, mu, logvar):
        recon_loss = F.mse_loss(x_hat, emb)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def process_batch(self, batch):
        emb, y = self.gcn.get_emb(batch)
        x_hat, mu, logvar = self.vae(emb, y)
        return emb, y, x_hat, mu, logvar

    def training_step(self, batch, batch_idx):
        emb, y, x_hat, mu, logvar = self.process_batch(batch)
        loss = self.compute_loss(emb, y, x_hat, mu, logvar)
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        emb, y, x_hat, mu, logvar = self.process_batch(batch)
        loss = self.compute_loss(emb, y, x_hat, mu, logvar)
        self.log('val_loss', loss, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        emb, y, x_hat, mu, logvar = self.process_batch(batch)
        loss = self.compute_loss(emb, y, x_hat, mu, logvar)
        self.log('test_loss', loss, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        
    def generate(self, num_samples, label_list, gcn):
        with torch.no_grad():
            all_z = torch.randn(num_samples * len(label_list), self.vae.latent_dim)
            all_labels = torch.tensor(label_list * num_samples, dtype=torch.long)
            all_label_embeddings = self.vae.label_embedding(all_labels)
            all_z_with_labels = torch.cat([all_z, all_label_embeddings], dim=1)
            all_samples = self.vae.decoder(all_z_with_labels)
            probabilities = F.softmax(gcn.fc_layers(all_samples), dim=1)
            _, all_indices = torch.max(probabilities, dim=1)
            shuffled_indices = torch.randperm(all_samples.size(0))
            return TensorDataset(all_samples[shuffled_indices], all_indices[shuffled_indices])
