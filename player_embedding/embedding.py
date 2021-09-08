import torch

import pandas as pd
import numpy as np

def generate_all_embedding(num_classes, hidden_dim, model, pc_loader):
    with torch.no_grad():
        model.eval()
        embeddings = torch.zeros(0, hidden_dim).cuda()
        classes = torch.zeros(0).long().cuda()

        for target, target_len, opponent, opponent_len, labels in pc_loader:
            target = target.cuda()
            opponent = opponent.cuda()

            labels = labels.squeeze(1).long().cuda()
            
            output = model(target, target_len, opponent, opponent_len)

            embeddings = torch.cat([embeddings, output], axis=0)
            classes = torch.cat([classes, labels], axis=0)
            
    return embeddings, classes.cpu().numpy()

def generate_class_embedding(num_classes, hidden_dim, model, pc_loader):
    with torch.no_grad():
        model.eval()
        class_embedding = torch.zeros(num_classes, hidden_dim).cuda()
        class_count = torch.zeros(num_classes).cuda()

        for target, target_len, opponent, opponent_len, labels in pc_loader:
            target = target.cuda()
            opponent = opponent.cuda()
            
            labels = labels.squeeze(1).long().cuda()
            output = model(target, target_len, opponent, opponent_len)

            scatter = torch.zeros(num_classes, labels.shape[0]).cuda()
            scatter[labels, torch.tensor([i for i in range(labels.shape[0])])] = 1

            class_count += scatter.sum(axis=1)

            class_embedding += torch.matmul(scatter, output)

        class_embedding /= class_count.unsqueeze(1).expand(num_classes, hidden_dim)
    
    return class_embedding