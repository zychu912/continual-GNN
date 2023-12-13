import torch
from torch_geometric.loader import DataLoader

class TaskManager:
    def __init__(self, dataset, batch_size=64, num_tasks=5, num_workers=1, train_ratio=0.6, val_ratio=0.2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.task_labels = torch.chunk(torch.randperm(torch.unique(dataset.y).max().item() + 1), self.num_tasks)
        
    def generate_dataloaders(self, task_id):
        task_mask = torch.where((self.dataset.y == self.task_labels[task_id].unsqueeze(1)).any(0))[0]
        random_task_mask = task_mask[torch.randperm(task_mask.size(0))]
        num_samples = random_task_mask.size(0)
        train_end = int(num_samples * self.train_ratio)
        val_end = train_end + int(num_samples * self.val_ratio)
        train_loader = DataLoader(self.dataset[random_task_mask[:train_end]], 
                                  batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        val_loader = DataLoader(self.dataset[random_task_mask[train_end:val_end]], 
                                  batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(self.dataset[random_task_mask[val_end:]], 
                                  batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader, test_loader
