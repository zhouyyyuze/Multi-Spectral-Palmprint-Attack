import my_multi_dataloader
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import My_DGCCA_PGD
from torchvision import transforms
import pytorch_ssim
import time
import DS_Proj_Matrix
from DGCCA_Train import DGCCA_train

torch.set_default_tensor_type(torch.DoubleTensor)


# Blue
mean_1 = [0.5071475, 0.5071475, 0.5071475]
std_1 = [0.05167011, 0.05167011, 0.05167011]

# Green
mean_2 = [0.43933, 0.43933, 0.43933]
std_2 = [0.041091993, 0.041091993, 0.041091993]

# NIR
mean_3 = [0.57223487, 0.57223487, 0.57223487]
std_3 = [0.024902835, 0.024902835, 0.024902835]

# Normalize
tf_1 = transforms.Compose([transforms.Normalize(mean=mean_1, std=std_1)])
tf_2 = transforms.Compose([transforms.Normalize(mean=mean_2, std=std_2)])
tf_3 = transforms.Compose([transforms.Normalize(mean=mean_3, std=std_3)])


def get_GDS_advnoise_v2(data, proj_matrix, k=50, eps=5):
    adv_noise = []
    for x in data:
        z = x @ (proj_matrix.T)
        for i in range(z.shape[0]):
            if i > k:
                z[i] = i * 0.02 * z[i]
            else:
                z[i] = z[i]
        x_ = z @ proj_matrix
        p = (x_ - x) / np.linalg.norm(x_ - x, ord=2, axis=None, keepdims=True)
        adv_noise.append(p * eps)
    adv_noise = np.array(adv_noise)
    adv_noise = np.clip(adv_noise, 0, 1)  # 限制在(0-1)
    return adv_noise

# DCCA结合GDS的, 返回的是对抗样本
def DGCCA_GDS_Attack(DGCCA_model, device, data_loader, proj_blue, proj_green, proj_nir, steps=100, k=50, alpha=3 / 255, eps_DGCCA=0.07, eps_GDS=5):
    DGCCA_PGD = My_DGCCA_PGD.DGCCA_PGD(model=DGCCA_model, eps=eps_DGCCA, alpha=alpha, steps=steps, random_start=False)
    DGCCA_model.eval()

    adv_DGCCA_images_1 = []
    adv_DGCCA_images_2 = []
    adv_DGCCA_images_3 = []

    images_1 = []
    images_2 = []
    images_3 = []

    for iteration, batch in enumerate(data_loader):
        image_1 = batch[0].to(device)
        image_2 = batch[2].to(device)
        image_3 = batch[4].to(device)

        if eps_DGCCA > 0:
            print("eps_DCCA > 0 start generate DGCCA perturbations")
            adv_DCCA_image_1, adv_DCCA_image_2, adv_DCCA_image_3 = DGCCA_PGD.get_adv_images(image_1, image_2, image_3)
        else:
            adv_DCCA_image_1 = image_1.reshape(image_1.shape[0], -1)
            adv_DCCA_image_2 = image_2.reshape(image_2.shape[0], -1)
            adv_DCCA_image_3 = image_3.reshape(image_3.shape[0], -1)

        adv_DGCCA_images_1.append(adv_DCCA_image_1)
        adv_DGCCA_images_2.append(adv_DCCA_image_2)
        adv_DGCCA_images_3.append(adv_DCCA_image_3)

        images_1.append(image_1.reshape(image_1.shape[0], -1))
        images_2.append(image_2.reshape(image_2.shape[0], -1))
        images_3.append(image_3.reshape(image_3.shape[0], -1))

    adv_DGCCA_images_1 = torch.cat(adv_DGCCA_images_1, dim=0).detach().cpu().numpy()
    adv_DGCCA_images_2 = torch.cat(adv_DGCCA_images_2, dim=0).detach().cpu().numpy()
    adv_DGCCA_images_3 = torch.cat(adv_DGCCA_images_3, dim=0).detach().cpu().numpy()

    images_1 = torch.cat(images_1, dim=0).detach().cpu().numpy()
    images_2 = torch.cat(images_2, dim=0).detach().cpu().numpy()
    images_3 = torch.cat(images_3, dim=0).detach().cpu().numpy()

    if eps_GDS > 0:
        print("eps_GDS>0")
        GDS_noise_1 = get_GDS_advnoise_v2(images_1, proj_blue, k, eps_GDS)
        GDS_noise_2 = get_GDS_advnoise_v2(images_2, proj_green, k, eps_GDS)
        GDS_noise_3 = get_GDS_advnoise_v2(images_3, proj_nir, k, eps_GDS)

        adv_images_1 = np.clip(adv_DGCCA_images_1 + GDS_noise_1, 0, 1)
        adv_images_2 = np.clip(adv_DGCCA_images_2 + GDS_noise_2, 0, 1)
        adv_images_3 = np.clip(adv_DGCCA_images_3 + GDS_noise_3, 0, 1)

    else:
        adv_images_1 = np.clip(adv_DGCCA_images_1, 0, 1)
        adv_images_2 = np.clip(adv_DGCCA_images_2, 0, 1)
        adv_images_3 = np.clip(adv_DGCCA_images_3, 0, 1)

    return adv_images_1, adv_images_2, adv_images_3


def plot_images(pictures, title):
    n_cols = 6
    n_rows = 6
    cnt = 1

    plt.figure(dpi=200, figsize=(n_rows, n_cols))
    tf = transforms.Compose([transforms.ToPILImage()])

    for i in range(n_cols):
        for j in range(n_rows):
            plt.subplot(n_cols, n_rows, cnt)
            image = torch.from_numpy(pictures[cnt - 1].reshape(3, 32, 32))
            plt.xticks(fontsize=2)
            plt.yticks(fontsize=2)
            plt.imshow(tf(image))
            cnt += 1
    plt.savefig(title)
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    print("Using", torch.cuda.device_count(), "GPUs")

    # parameter
    batch_size = 240
    resize = 32
    # Todo path
    root = 'data'
    filename_B = 'B_img.csv'
    filename_G = 'G_img.csv'
    filename_NIR = 'NIR_img.csv'

    hand_dataset_train = my_multi_dataloader.MyDataset(root=root, filename_1=filename_B,
                                                      filename_2=filename_G,
                                                      filename_3=filename_NIR,
                                                      resize=32, mode='train')
    hand_loader_train = DataLoader(hand_dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)

    hand_dataset_test = my_multi_dataloader.MyDataset(root=root, filename_1=filename_B,
                                                      filename_2=filename_G,
                                                      filename_3=filename_NIR,
                                                      resize=32, mode='test')
    hand_loader_test = DataLoader(hand_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # train DGCCA model
    # DGCCA_model = DGCCA_train.train_DGCCA(hand_loader_train, hand_loader_test, epoch_num=10)

    # load DGCCA model
    DGCCA_model = torch.load('save_model/DGCCA.pt')

    # generate GDS projection matrix
    # proj_D500_blue, proj_D500_green, proj_D500_nir = DS_Proj_Matrix.generate_proj_matrix(hand_loader_train)
    # np.save("save_model/proj_D500_blue", proj_D500_blue)  # Blue
    # np.save("save_model/proj_D500_green", proj_D500_green)  # Green
    # np.save("save_model/proj_D500_nir", proj_D500_nir)  # NIR

    # load GDS projection matrix
    proj_D500_blue = np.load("save_model/proj_D500_blue.npy")  # projection matrix blue
    proj_D500_green = np.load("save_model/proj_D500_green.npy")  # projection matrix green
    proj_D500_nir = np.load("save_model/proj_D500_nir.npy")  # projection matrix nir


    # ================================ start generating adversarial examples ====================================
    # star time
    timestart = time.time()
    adv_outputs_1, adv_outputs_2, adv_outputs_3 = DGCCA_GDS_Attack(
        DGCCA_model=DGCCA_model,
        device=device,
        data_loader=hand_loader_test,
        proj_blue=proj_D500_blue[:125],
        proj_green=proj_D500_green[:125],
        proj_nir=proj_D500_nir[:125],
        steps=20, k=0, alpha=60 / 255,
        eps_DGCCA=0.002,
        eps_GDS=2)

    # end time
    timeend = time.time()
    print("Took", timeend - timestart, "seconds to run")
    # ======================================= save data ======================================

    np.save("save_data/adv_test_blue.npy", adv_outputs_1)
    np.save("save_data/adv_test_green.npy", adv_outputs_2)
    np.save("save_data/adv_test_nir.npy", adv_outputs_3)

    adv_test_blue = np.load("save_data/adv_test_blue.npy")
    adv_test_green = np.load("save_data/adv_test_green.npy")
    adv_test_nir = np.load("save_data/adv_test_nir.npy")

    plot_images(adv_test_blue, "adv_blue")
    plot_images(adv_test_green, "adv_green")
    plot_images(adv_test_nir, "adv_nir")