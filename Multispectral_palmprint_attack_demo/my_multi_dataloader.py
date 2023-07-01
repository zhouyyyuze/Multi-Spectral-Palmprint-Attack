# 导入相关库
import csv
import random
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyDataset(Dataset):

    def __init__(self, root, filename_1, filename_2, filename_3, resize, mode):
        super(MyDataset, self).__init__()
        self.root = root
        self.resize = resize
        self.mode = mode
        self.images_1, self.labels_1 = self.load_csv(filename_1)  
        self.images_2, self.labels_2, = self.load_csv(filename_2) 
        self.images_3, self.labels_3, = self.load_csv(filename_3)

        # randnum = random.randint(0, 100)
        # print("randnum", randnum)

        randnum = 89

        random.seed(randnum)
        random.shuffle(self.images_1)
        random.seed(randnum)
        random.shuffle(self.labels_1)
        random.seed(randnum)
        random.shuffle(self.images_2)
        random.seed(randnum)
        random.shuffle(self.labels_2)
        random.seed(randnum)
        random.shuffle(self.images_3)
        random.seed(randnum)
        random.shuffle(self.labels_3)

        if self.mode == 'all':

            self.images_1 = self.images_1
            self.labels_1 = self.labels_1

            self.images_2 = self.images_2
            self.labels_2 = self.labels_2

            self.images_3 = self.images_3
            self.labels_3 = self.labels_3

        elif self.mode == 'train':  # 80%
            self.images_1 = self.images_1[:int(0.8 * len(self.images_1))]
            self.labels_1 = self.labels_1[:int(0.8 * len(self.labels_1))]

            self.images_2 = self.images_2[:int(0.8 * len(self.images_2))]
            self.labels_2 = self.labels_2[:int(0.8 * len(self.labels_2))]

            self.images_3 = self.images_3[:int(0.8 * len(self.images_3))]
            self.labels_3 = self.labels_3[:int(0.8 * len(self.labels_3))]

        elif self.mode == 'test':  # 20%
            self.images_1 = self.images_1[int(0.8 * len(self.images_1)):]
            self.labels_1 = self.labels_1[int(0.8 * len(self.labels_1)):]

            self.images_2 = self.images_2[int(0.8 * len(self.images_2)):]
            self.labels_2 = self.labels_2[int(0.8 * len(self.labels_2)):]

            self.images_3 = self.images_3[int(0.8 * len(self.images_3)):]
            self.labels_3 = self.labels_3[int(0.8 * len(self.labels_3)):]


    def __len__(self):
        return len(self.images_1)

    def __getitem__(self, idx):

        img_1, label_1 = self.images_1[idx], self.labels_1[idx]
        img_2, label_2 = self.images_2[idx], self.labels_2[idx]
        img_3, label_3 = self.images_3[idx], self.labels_3[idx]

        tf_1 = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((int(self.resize), int(self.resize))),
            transforms.ToTensor(),
        ])

        tf_2 = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            # 这里开始读取了数据的内容了
            transforms.Resize((int(self.resize), int(self.resize))),
            transforms.ToTensor(),
        ])

        tf_3 = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((int(self.resize), int(self.resize))),
            transforms.ToTensor(),
        ])

        img_1 = tf_1(img_1)
        img_2 = tf_2(img_2)
        img_3 = tf_3(img_3)

        return img_1, label_1, img_2, label_2, img_3, label_3  # 返回当前的数据内容和标签

    def load_csv(self, filename):
        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels
