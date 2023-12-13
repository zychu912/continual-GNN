import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch_geometric.nn import GCNConv, global_mean_pool
import lightning.pytorch as pl

class GraphClassifier(nn.Module):
    def __init__(self, layers, num_classes=10, batch_norm=True):
        super().__init__()
        self.conv1 = GCNConv(layers[0], layers[1])
        self.bn1 = nn.BatchNorm1d(layers[1])
        modules = []
        for in_features, out_features in zip(layers[1:], layers[2:]):
            modules.append(GCNConv(in_features, out_features))
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_features))
            modules.append(nn.ReLU())
        self.network = nn.Sequential(*modules)
        self.fc_final = nn.Linear(layers[-1], num_classes)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        for module in self.network:
            if isinstance(module, GCNConv):
                x = F.relu(module(x, data.edge_index))
            else:
                x = module(x)
        x = global_mean_pool(x, data.batch)
        return F.log_softmax(self.fc_final(x), dim=1)

class GCNClassifier(pl.LightningModule):
    def __init__(self, layers, current_task, root, gamma, 
                 distillation_t=1, num_classes=10, batch_size=64, batch_norm=True, lr=0.0001):
        super().__init__()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.classifier = GraphClassifier(layers, num_classes, batch_norm)
        self.current_task = current_task
        self.root = root
        if self.current_task > 0:
            self.load_task_model(self.current_task - 1)
            self.teacher = copy.deepcopy(self.classifier)
        self.gamma = gamma
        self.distillation_t = distillation_t
        self.lr = lr
        self.current_task_losses = []
        self.previous_task_losses = []
        self.save_hyperparameters()
        self.automatic_optimization = False

    def forward(self, data):
        return self.classifier(data)

    def load_task_model(self, task_id):
        load_path = f'{self.root}task{task_id}.pth'
        self.classifier.load_state_dict(torch.load(load_path))

    # def load_task_model(self, task_id):
    #     load_path = f'{self.root}task{task_id}.pth'
    #     checkpoint = torch.load(load_path)
    #     self.classifier.load_state_dict(checkpoint['classifier'])

    # def save_task_model(self, task_id):
    #     save_path = f'{self.root}task{task_id}.pth'
    #     torch.save({'classifier': self.classifier.state_dict()}, save_path)

    def distillation_loss(self, student_outputs, teacher_outputs):
        teacher_probs = F.softmax(teacher_outputs / self.distillation_t, dim=1)
        student_log_probs = F.log_softmax(student_outputs / self.distillation_t, dim=1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        if self.current_task == 0:
            output = self(batch)
            total_loss = F.cross_entropy(output, batch.y)
        else:
            current_batch = batch['current']
            current_output = self(current_batch)
            
            previous_batch = batch['previous']
            student_output = self(previous_batch)
            
            self.teacher.eval()
            with torch.no_grad():
                teacher_output = self.teacher(previous_batch)
            
            current_loss = F.cross_entropy(current_output, current_batch.y)
            prev_loss = self.distillation_loss(student_output, teacher_output)
            total_loss = (1 - self.gamma) * current_loss + self.gamma * prev_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.log('batch_total_loss', total_loss)
        if self.current_task != 0:
            self.log('batch_current_loss', current_loss)
            self.log('batch_prev_loss', prev_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = F.cross_entropy(output, batch.y)
        acc = self.accuracy(output, batch.y)
        self.log_dict({'val_loss': loss,  'val_acc': acc}, batch_size=self.hparams.batch_size)

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = F.cross_entropy(output, batch.y)
        acc = self.accuracy(output, batch.y)
        self.log_dict({'test_loss': loss,  'test_acc': acc}, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
