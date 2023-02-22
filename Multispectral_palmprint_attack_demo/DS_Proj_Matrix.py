import numpy as np
import scipy.linalg
import torch
import scipy
from torch.utils.data import DataLoader
import my_multi_dataloader


def sort_dataset(datasets, labels):
    data_order = []
    for i in (np.unique(labels)):
        data_order.append(scipy.linalg.orth(datasets[labels == i].T, rcond=None))
    return data_order

def dataloader_to_numpy(data_loader):
    outputs1 = []
    outputs2 = []
    outputs3 = []
    labels1 = []
    labels2 = []
    labels3 = []
    for iteration, batch in enumerate(data_loader):
        batch_x1 = batch[0].reshape(240, -1).to(torch.double)
        batch_x2 = batch[2].reshape(240, -1).to(torch.double)
        batch_x3 = batch[4].reshape(240, -1).to(torch.double)
        label_1 = batch[1].reshape(-1)
        label_2 = batch[3].reshape(-1)
        label_3 = batch[5].reshape(-1)
        outputs1.append(batch_x1)
        outputs2.append(batch_x2)
        outputs3.append(batch_x3)
        labels1.append(label_1)
        labels2.append(label_2)
        labels3.append(label_3)
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy(),
                   torch.cat(outputs3, dim=0).cpu().numpy()]

        labels = [torch.cat(labels1, dim=0).cpu().numpy(),
                  torch.cat(labels2, dim=0).cpu().numpy(),
                  torch.cat(labels3, dim=0).cpu().numpy()]
    return outputs, labels


def get_proj_matrix(resize, datasets, dim=20):
    G = np.zeros([resize * resize * 3, resize * resize * 3, ])
    for X in datasets:
        proj_X = 0
        for i in range(X.shape[1]):
            proj_X += X[:, i:i + 1] @ X[:, i:i + 1].T
        G += proj_X
    eigenvalue, eigenvector = np.linalg.eig(G)
    proj_Dc = eigenvector.T[dim:]
    return proj_Dc.real


def generate_proj_matrix(dataloader):
    outputs, labels_all = dataloader_to_numpy(dataloader)
    all_blue = outputs[0]
    all_green = outputs[1]
    all_nir = outputs[2]

    all_blue_order = sort_dataset(all_blue, labels_all[0])
    all_green_order = sort_dataset(all_green, labels_all[0])
    all_nir_order = sort_dataset(all_nir, labels_all[0])

    proj_D500_blue = get_proj_matrix(32, all_blue_order, dim=0)
    proj_D500_green = get_proj_matrix(32, all_green_order, dim=0)
    proj_D500_nir = get_proj_matrix(32, all_nir_order, dim=0)

    return proj_D500_blue, proj_D500_green, proj_D500_nir


if __name__ == '__main__':
    '''Parameters Section 参数选择'''
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    print("Using", torch.cuda.device_count(), "GPUs")
    # the path to save the final learned features

    # 超参数设置
    batch_size = 12
    resize = 32  # 修改图片尺寸

    '''end of parameters section'''

    # Todo 修改路径
    root = 'data'
    filename_B = 'B_img.csv'
    filename_G = 'G_img.csv'
    filename_NIR = 'NIR_img.csv'

    # 总数据集
    dataset = my_multi_dataloader.MyDataset(root=root, filename_1=filename_B,
                                            filename_2=filename_G,
                                            filename_3=filename_NIR,
                                            resize=resize, mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    proj_D500_blue, proj_D500_green, proj_D500_nir = generate_proj_matrix(dataloader)

    np.save("save_model/proj_D500_blue", proj_D500_blue)  # Blue
    np.save("save_model/proj_D500_green", proj_D500_green)  # Green
    np.save("save_model/proj_D500_nir", proj_D500_nir)  # NIR
