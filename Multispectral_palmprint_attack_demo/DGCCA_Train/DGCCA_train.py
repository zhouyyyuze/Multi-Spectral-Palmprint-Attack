## similar to github.com/Michaelvll/DeepCCA main
import torch
from DGCCA_Train.models import DeepGCCA
import time
import logging
from torch.utils.data import DataLoader

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)


class Solver():
    def __init__(self, model, outdim_size, epoch_num, batch_size, learning_rate, reg_par,
                 device=torch.device('cpu')):
        self.model = model  # nn.DataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.Adam(
            self.model.model_list.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.7)
        self.device = device

        self.outdim_size = outdim_size

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("DCCA.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, train_loader, test_loader=None, checkpoint='checkpoint.model'):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        train_losses = []

        best_test_loss = 0

        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()

            for step, batch in enumerate(train_loader):
                self.optimizer.zero_grad()

                batch_size = batch[0].shape[0]
                batch_x1 = batch[0].reshape(batch_size, -1).to(torch.double).to(self.device)
                batch_x2 = batch[2].reshape(batch_size, -1).to(torch.double).to(self.device)
                batch_x3 = batch[4].reshape(batch_size, -1).to(torch.double).to(self.device)

                batch_x = [batch_x1, batch_x2, batch_x3]

                output = self.model(batch_x)
                loss = self.loss(output)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.mean(train_losses)

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if test_loader is not None:
                with torch.no_grad():
                    self.model.eval()
                    test_loss, _, _ = self.test(test_loader)
                    info_string += " - val_loss: {:.4f}".format(test_loss)
                    if test_loss < best_test_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1,
                                                                                                             best_test_loss,
                                                                                                             test_loss,
                                                                                                             checkpoint))
                        best_test_loss = test_loss
                        torch.save(self.model.state_dict(), checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_test_loss))
            else:
                torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        if test_loader is not None:
            loss, _, _ = self.test(test_loader)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

    def test(self, data_loader):
        with torch.no_grad():
            losses, outputs_list, labels = self._get_outputs(data_loader)
            return np.mean(losses), outputs_list, labels

    def _get_outputs(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            losses = []
            outputs_1 = []
            outputs_2 = []
            outputs_3 = []
            labels_1 = []
            labels_2 = []
            labels_3 = []

            for step, batch in enumerate(data_loader):
                batch_size = batch[0].shape[0]
                batch_x1 = batch[0].reshape(batch_size, -1).to(torch.double).to(self.device)
                batch_x2 = batch[2].reshape(batch_size, -1).to(torch.double).to(self.device)
                batch_x3 = batch[4].reshape(batch_size, -1).to(torch.double).to(self.device)
                label_1 = batch[1].reshape(-1).to(self.device)
                label_2 = batch[3].reshape(-1).to(self.device)
                label_3 = batch[5].reshape(-1).to(self.device)

                outputs = self.model([batch_x1, batch_x2, batch_x3])

                outputs_1.append(outputs[0])
                outputs_2.append(outputs[1])
                outputs_3.append(outputs[2])
                labels_1.append(label_1)
                labels_2.append(label_2)
                labels_3.append(label_3)

                loss = self.loss(outputs)
                losses.append(loss.item())

                outputs_final = [torch.cat(outputs_1, dim=0).cpu().numpy(),
                                 torch.cat(outputs_2, dim=0).cpu().numpy(),
                                 torch.cat(outputs_3, dim=0).cpu().numpy()]

                labels = [torch.cat(labels_1, dim=0).cpu().numpy(),
                          torch.cat(labels_2, dim=0).cpu().numpy(),
                          torch.cat(labels_3, dim=0).cpu().numpy()]

        return losses, outputs_final, labels

    def save(self, name):
        torch.save(self.model, name)


def train_DGCCA(train_loader, test_loader, learning_rate=1e-3, epoch_num=20, reg_par=1e-5, batch_size=240,
                use_all_singular_values=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", torch.cuda.device_count(), "GPUs")
    # the path to save the final learned features
    save_name = './DGCCA.model'
    # the size of the new space learned by the model (number of the new features)
    outdim_size = 256
    # number of layers with nodes in each one
    layer_sizes1 = [1024, 1024, outdim_size]
    layer_sizes2 = [1024, 1024, outdim_size]
    layer_sizes3 = [1024, 1024, outdim_size]

    layer_sizes_list = [layer_sizes1, layer_sizes2, layer_sizes3]
    # size of the input for view 1, view 2 and view 3
    input_shape_list = [3072, 3072, 3072]
    print(input_shape_list)

    # the parameters for training the network
    use_all_singular_values = False

    # Building, training, and producing the new features by DCCA
    model = DeepGCCA(layer_sizes_list, input_shape_list, outdim_size,
                     use_all_singular_values, device=device).double()

    solver = Solver(model, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par, device=device)

    solver.fit(train_loader=train_loader, test_loader=test_loader, checkpoint=save_name)

    return model

    torch.save(model, 'save_model/DGCCA.pt')


if __name__ == '__main__':
    # Todo path
    root = '..\data'
    filename_B = 'B_img.csv'
    filename_G = 'G_img.csv'
    filename_NIR = 'NIR_img.csv'

    dataset_train = my_multi_dataloader.MyDataset(root=root, filename_1=filename_B, filename_2=filename_G,
                                                  filename_3=filename_NIR,
                                                  resize=32, mode='train')
    loader_train = DataLoader(dataset_train, batch_size=240, shuffle=False, num_workers=0)

    dataset_test = my_multi_dataloader.MyDataset(root=root, filename_1=filename_B, filename_2=filename_G,
                                                 filename_3=filename_NIR,
                                                 resize=32, mode='test')
    loader_test = DataLoader(dataset_test, batch_size=240, shuffle=False, num_workers=0)

    train_DGCCA(loader_train, loader_test)
