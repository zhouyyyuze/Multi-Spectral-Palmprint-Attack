# 导入相关库
import csv
import random
import os
import glob
# 数据地址，注意是分类文件夹的上层文件夹
# dataset文件夹中的两个子文件夹分别对应两个类名

def create_csv(root, path, save_name):

    # root = 'E:\Python_WorkSpace\Multispectral_Palmprint_Attack_3views\zCompCode\data'
    #
    # path = os.path.join(root, "Multispectral_G")
    # save_name = 'G_img.csv'

    # 对数据地址中的文件夹进行遍历，将类名存放于列表names中
    names = os.listdir(path)
    # 创建名为images的空列表用于存放图像地址
    images = []
    # 创建名称、标签字典，用于存放类名和标签
    names_labels = {}


    # 遍历类名
    for name in names:
        # 遍历第一个类名的时候，这个时候仅有一个键，为ants_image
        # 遍历第二个类名的时候，这个时候有两个键，分别为ants_image和bees_image
        # 那么字典names_label的键的长度在第一次遍历的时候为0，第二次遍历的时候为1
        names_labels[name] = len(names_labels.keys())
        # glob.glob返回所有匹配的文件路径列表。
        # 相对于本文而言，返回的是路径dataset/ants_image/xx...xxx.jpg的文件
        # 用*号匹配复杂没有规律的所有格式为jpg的图片名，且*表示匹配0个或多个字符
        # images += glob.glob(os.path.join(root, name, '*.bmp'))
        f = glob.iglob(os.path.join(path, name, '*.bmp'))
        count = 0
        for py in f:
            # if count == 8:
            #     break
            images.append(py)
            count = count + 1
        # images += glob.glob(os.path.join(root, name, '*.bmp'))
        # print(os.path.join(root, name, '*.jpg'))

    print(images)

    # 对csv文件进行写操作，如果没有csv文件会自动创建
    # (1)图中，逗号前为文件名，逗号后为标签
    # newline=''的作用是防止每一行数据后面都自动增加了一个空行

    with open(os.path.join(root, save_name ), mode='w', newline='') as f:
        writer = csv.writer(f)
        for img in images:
            # 用os.sep切割具有通用性，自动识别路径分隔符windows和linus
            name = img.split(os.sep)[-2]
            # print(name)
            label = names_labels[name]
            writer.writerow([img, label])

if __name__ == '__main__':

    root = 'E:\Python_WorkSpace\Multispectral_palmprint_attack_demo\data'
    path = os.path.join(root, "Multispectral_B")
    save_name = 'B_img.csv'
    create_csv(root, path, save_name)

    root = 'E:\Python_WorkSpace\Multispectral_palmprint_attack_demo\data'
    path = os.path.join(root, "Multispectral_G")
    save_name = 'G_img.csv'
    create_csv(root, path, save_name)

    root = 'E:\Python_WorkSpace\Multispectral_palmprint_attack_demo\data'
    path = os.path.join(root, "Multispectral_NIR")
    save_name = 'NIR_img.csv'
    create_csv(root, path, save_name)





