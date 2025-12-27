import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from chen2026spfere.utils.server_logger import fl_logger

def extract_targets(ds):
    # 1) indices
    if isinstance(ds, Subset):
        parent_targets = extract_targets(ds.dataset)
        return np.asarray(parent_targets)[ds.indices]

    # 2) ConcatDataset
    if isinstance(ds, ConcatDataset):
        parts = [extract_targets(d) for d in ds.datasets]
        return np.concatenate(parts)

    # 3) targets
    if hasattr(ds, "targets") and ds.targets is not None:
        return np.asarray(ds.targets)

    # 4) STL10/EuroSAT
    if hasattr(ds, "labels") and ds.labels is not None:
        return np.asarray(ds.labels)

    # 5) ImageFolder
    if hasattr(ds, "imgs"):
        return np.asarray([y for _, y in ds.imgs])
    if hasattr(ds, "samples"):
        return np.asarray([y for _, y in ds.samples])

    raise AttributeError(f"Cannot extract targets from dataset type: {type(ds)}")

def make_calib_loader(base_dataset, num_samples=2000, batch_size=256, seed=2025):
    N = len(base_dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(num_samples, N), replace=False)
    calib_subset = Subset(base_dataset, idx)
    return DataLoader(calib_subset, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True, drop_last=False)

def avg_load_data(data="cifar10", batch_size=128, ratio=0.1, seed=42, server_logger_filename=None):
    np.random.seed(seed)
    data = data.lower()

    if data == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.247, 0.243, 0.261])
        ])
        test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    elif data == "svhn":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                 std=[0.198, 0.201, 0.197])
        ])
        test_set = datasets.SVHN(root="data", split='test', download=True, transform=transform)
        test_set.targets = test_set.labels

    elif data == "f_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        test_set = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

    elif data == "stl_10":
        from torchvision.datasets import STL10

        mean = [0.4408, 0.4279, 0.3867]
        std  = [0.2682, 0.2610, 0.2686]

        train_tf = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_set  = STL10(root="data", split="test",  download=True, transform=test_tf)

    elif data == "eurosat_rgb":
        from torch.utils.data import random_split

        mean = [0.3401, 0.3807, 0.4079]
        std  = [0.1344, 0.1260, 0.1251]

        train_tf = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_ratio = 0.8
        from torchvision.datasets import EuroSAT
        full = EuroSAT(root="data", download=True, transform=train_tf)
        n_train = int(len(full) * train_ratio)
        n_test = len(full) - n_train
        _, test_set = random_split(full, [n_train, n_test], generator=torch.Generator().manual_seed(42))
        test_set.dataset.transform = test_tf

    else:
        raise ValueError(f"Unsupported dataset: {data}")
    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=32, drop_last=False)
    calib_loader = make_calib_loader(test_set, num_samples=2000, batch_size=batch_size)

    fl_logger(f"[+] {data.upper()} test dataset loaded with ratio [{ratio}].", server_logger_filename)

    return test_loader, calib_loader

def split_data_dirichlet(dataset, ratio=0.1, alpha=0.5, replace=False, seed=42, server_logger_filename=None):
    num_samples = int(len(dataset) * ratio)
    labels = extract_targets(dataset)
    num_classes = len(np.unique(labels))
    np.random.seed(seed)
    class_priors = np.random.dirichlet([alpha] * num_classes)

    client_indices = []
    remaining_samples = num_samples

    for c in range(num_classes):
        indices = np.where(labels == c)[0]
        if c < num_classes - 1:
            num_class_samples = max(1, int(class_priors[c] * num_samples)-num_classes)
        else:
            num_class_samples = remaining_samples
        if replace == False:
            num_class_samples = min(len(indices), num_class_samples)
        selected_indices = np.random.choice(indices, num_class_samples, replace=replace)
        client_indices.extend(selected_indices)
        remaining_samples -= num_class_samples

    np.random.shuffle(client_indices)
    client_indices = client_indices[:num_samples]

    return Subset(dataset, client_indices)

def dirichlet_load_data(data="cifar10", batch_size=128, ratio=0.1, alpha=0.5, seed=42, server_logger_filename=None):
    data = data.lower()
    
    if data == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])
        test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    elif data == "svhn":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.198, 0.201, 0.197])
        ])
        test_set = datasets.SVHN(root="data", split='test', download=True, transform=transform)
        test_set.targets = test_set.labels

    elif data == "f_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        test_set = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
        
    elif data == "stl_10":
        from torchvision.datasets import STL10

        mean = [0.4408, 0.4279, 0.3867]
        std  = [0.2682, 0.2610, 0.2686]

        train_tf = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        test_set  = STL10(root="data", split="test",  download=True, transform=test_tf)

    elif data == "eurosat_rgb":
        from torch.utils.data import random_split

        mean = [0.3401, 0.3807, 0.4079]
        std  = [0.1344, 0.1260, 0.1251]

        train_tf = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_ratio = 0.8
        from torchvision.datasets import EuroSAT
        full = EuroSAT(root="data", download=True, transform=train_tf)
        n_train = int(len(full) * train_ratio)
        n_test = len(full) - n_train
        _, test_set = random_split(full, [n_train, n_test], generator=torch.Generator().manual_seed(42))
        test_set.dataset.transform = test_tf

    else:
        raise ValueError(f"Unknown dataset: {data}")
    
    subset_test_set = split_data_dirichlet(test_set, alpha=alpha, ratio=1.0, replace=True, seed=seed, server_logger_filename=server_logger_filename)
    test_loader = DataLoader(subset_test_set, batch_size=batch_size, shuffle=False, num_workers=32, drop_last=False)
    calib_loader = make_calib_loader(test_set, num_samples=2000, batch_size=batch_size)
    fl_logger(f"[+] {data.upper()} test dataset loaded with ratio [{ratio}] and Dirichlet alpha [{alpha}].", server_logger_filename)

    return test_loader, calib_loader

def load_data(data = "cifar10", batch_size = 128, ratio = 1, alpha = 10, seed = 42, mode = "avg", server_logger_filename = None):
    if mode == "dirichlet":
        test_loader, calib_loader = dirichlet_load_data(data, batch_size, ratio, alpha, seed=seed, server_logger_filename=server_logger_filename)
    else:
        test_loader, calib_loader = avg_load_data(data, batch_size, ratio, seed=seed, server_logger_filename=server_logger_filename)
    return test_loader, calib_loader