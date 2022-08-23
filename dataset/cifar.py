import os

import torch
from torchvision import transforms, datasets


def get_data_folder(opt):
    """
    return server-dependent path to store the data
    """
    root_path = '/home/lab265/lab265/datasets'
    data_folder = os.path.join(root_path, opt.dataset)

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


def getDataLoader(opt):
    # data_folder
    data_folder = get_data_folder(opt)

    if opt.dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    else:
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    # train_transform
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    # test_transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    # dataset
    if opt.dataset == 'CIFAR10':
        num_classes = 10
        train_dataset = datasets.CIFAR10(root=data_folder, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(root=data_folder, train=False, transform=test_transform, download=True)
    elif opt.dataset == 'CIFAR100':
        num_classes = 100
        train_dataset = datasets.CIFAR100(root=data_folder, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(root=data_folder, train=False, transform=test_transform, download=True)
    else:
        assert False
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size,
                                               shuffle=True, pin_memory=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size,
                                              shuffle=False, pin_memory=True, num_workers=8)

    return train_loader, test_loader, num_classes
