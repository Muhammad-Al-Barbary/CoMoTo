import torch
from torch.nn import KLDivLoss
from torch.nn.functional import softmax, log_softmax, cosine_similarity

def KD_loss(student_outputs, teacher_outputs, alpha = 1, T = 1):
    student_outputs = log_softmax(student_outputs/T, dim=1)
    teacher_outputs = softmax(teacher_outputs/T, dim=1)
    loss =  KLDivLoss(reduction='batchmean')(student_outputs, teacher_outputs)
    loss = (alpha * T * T) * loss
    return loss


def ImPA_loss(positive_features, negative_features, beta = 1):
    positive_features = positive_features.view(positive_features.size(0), -1)  
    negative_features = negative_features.view(negative_features.size(0), -1)
    pos_similarity = cosine_similarity(positive_features.unsqueeze(1), positive_features.unsqueeze(0), dim=2)
    neg_similarity = cosine_similarity(positive_features.unsqueeze(1), negative_features.unsqueeze(0), dim=2)
    loss = torch.mean(torch.relu(1 - pos_similarity)) + torch.mean(torch.relu(neg_similarity))
    loss = beta * loss
    return loss