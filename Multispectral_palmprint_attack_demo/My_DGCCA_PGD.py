import torch
import torch.nn as nn
from torchvision import transforms


class DGCCA_PGD():
    def __init__(self, model, eps=0.3, alpha=2 / 255, steps=40, random_start=True):
        self.device = next(model.parameters()).device  # 获取device
        self.model = nn.DataParallel(model)
        self.name = 'PGD'
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss = model.loss
        self.random_start = random_start
        self._targeted = False


        mean_1 = [0.5071475, 0.5071475, 0.5071475]
        std_1 = [0.05167011, 0.05167011, 0.05167011]

        mean_2 = [0.43933, 0.43933, 0.43933]
        std_2 = [0.041091993, 0.041091993, 0.041091993]

        mean_3 = [0.57223487, 0.57223487, 0.57223487]
        std_3 = [0.024902835, 0.024902835, 0.024902835]

        self.tf_1 = transforms.Compose([transforms.Normalize(mean=mean_1, std=std_1)])
        self.tf_2 = transforms.Compose([transforms.Normalize(mean=mean_2, std=std_2)])
        self.tf_3 = transforms.Compose([transforms.Normalize(mean=mean_3, std=std_3)])
        print('eps:', self.eps)

    def get_adv_images(self, images_1, images_2, images_3):
        # .to(self.device)
        images_1 = images_1.clone().detach().to(self.device)
        images_2 = images_2.clone().detach().to(self.device)
        images_3 = images_3.clone().detach().to(self.device)

        adv_images_1 = images_1.clone().detach()
        adv_images_2 = images_2.clone().detach()
        adv_images_3 = images_3.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images_1 = adv_images_1 + torch.empty_like(adv_images_1).uniform_(-self.eps, self.eps)
            adv_images_2 = adv_images_2 + torch.empty_like(adv_images_2).uniform_(-self.eps, self.eps)
            adv_images_3 = adv_images_3 + torch.empty_like(adv_images_3).uniform_(-self.eps, self.eps)

            adv_images_1 = torch.clamp(adv_images_1, min=-1, max=1).detach()
            adv_images_2 = torch.clamp(adv_images_2, min=-1, max=1).detach()
            adv_images_3 = torch.clamp(adv_images_3, min=-1, max=1).detach()

        for _ in range(self.steps):
            adv_images_1.requires_grad = True
            adv_images_2.requires_grad = True
            adv_images_3.requires_grad = True

            # Calculate loss
            [o1, o2, o3] = self.model([self.tf_1(adv_images_1).reshape(adv_images_1.shape[0], -1),
                                       self.tf_2(adv_images_2).reshape(adv_images_2.shape[0], -1),
                                       self.tf_3(adv_images_3).reshape(adv_images_3.shape[0], -1)])

            cost = self.loss([o1, o2, o3])

            # Update adversarial images
            grad_1 = torch.autograd.grad(cost, adv_images_1, retain_graph=True, create_graph=False)[0]
            grad_2 = torch.autograd.grad(cost, adv_images_2, retain_graph=True, create_graph=False)[0]
            grad_3 = torch.autograd.grad(cost, adv_images_3, retain_graph=False, create_graph=False)[0]

            adv_images_1 = adv_images_1.detach() + self.alpha * grad_1.sign()
            adv_images_2 = adv_images_2.detach() + self.alpha * grad_2.sign()
            adv_images_3 = adv_images_3.detach() + self.alpha * grad_3.sign()

            delta_1 = torch.clamp(adv_images_1 - images_1, min=-self.eps, max=self.eps)
            delta_2 = torch.clamp(adv_images_2 - images_2, min=-self.eps, max=self.eps)
            delta_3 = torch.clamp(adv_images_3 - images_3, min=-self.eps, max=self.eps)

            adv_images_1 = torch.clamp(images_1 + delta_1, min=0, max=1).detach()
            adv_images_2 = torch.clamp(images_2 + delta_2, min=0, max=1).detach()
            adv_images_3 = torch.clamp(images_3 + delta_3, min=0, max=1).detach()

        # return adv_images_1, adv_images_2
        return adv_images_1.reshape(adv_images_1.shape[0], -1), \
               adv_images_2.reshape(adv_images_2.shape[0], -1), \
               adv_images_3.reshape(adv_images_3.shape[0], -1)
