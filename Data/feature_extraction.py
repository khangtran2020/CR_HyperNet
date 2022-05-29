import pickle
import os
import numpy as np
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def get_feat(imgs):
    model = models.resnet18(pretrained=True)
    model.fc = Identity()
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images = normalize(torch.from_numpy(imgs))
    embeddings = model(images)
    return embeddings.detach().numpy()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def feature_extraction_cifar10():
    file_path = "cifar10/cifar-10-batches-py"
    train_batch = [ 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_batch = 'test_batch'

    # keys b'batch_label', b'labels', b'data', b'filenames'
    # working on test batch
    test_dict = unpickle(os.path.join(file_path,test_batch))
    test_data = test_dict[b'data']
    test_label = np.array(test_dict[b'labels'])
    test_data = test_data.reshape((test_data.shape[0], 3, 32, 32)).astype(np.float32)
    test_embeddings = get_feat(test_data)

    train_embeddings = None
    train_label = None
    for i, file in enumerate(train_batch):
        temp_dict = unpickle(os.path.join(file_path,file))
        temp_data = temp_dict[b'data']
        temp_label = np.array(temp_dict[b'labels'])
        temp_data = temp_data.reshape((temp_data.shape[0], 3, 32, 32)).astype(np.float32)
        temp_embeddings = get_feat(temp_data)
        if i == 0:
            train_embeddings = temp_embeddings
            train_label = temp_label
        else:
            train_embeddings = np.concatenate((train_embeddings, temp_embeddings), axis=0)
            train_label = np.concatenate((train_label, temp_label), axis=0)
    return train_embeddings, train_label, test_embeddings, test_label


#
# train_emb, train_lab, test_emb, test_lab = feature_extraction_cifar10()
# print(train_emb.shape, train_lab.shape, test_emb.shape, test_lab.shape)