import torch.utils.data
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as T
import numpy as np

# Classe accessible à tout le fichier
class RotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return F.rotate(x, self.angle, fill=(0,))


class RotatedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, download=True, num_tasks=5, per_task_rotation=45):
        self.dataset = torchvision.datasets.MNIST(root=root, train=train, transform=transform, download=download)
        self.transform = transform
        self.rotation_angles = [float(task * per_task_rotation) for task in range(num_tasks)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        angle = np.random.choice(self.rotation_angles)
        rotated_image = F.rotate(image, angle, fill=(0,))
        return rotated_image, label, angle


def flattened_rotMNIST(num_tasks, per_task_rotation, batch_size, transform=[]):
    g = torch.Generator()
    g.manual_seed(0)

    extended_transform = transform.copy()
    extended_transform.extend([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    composed_transform = T.Compose(extended_transform)

    train = RotatedMNISTDataset('./data/', train=True, download=True, transform=composed_transform, num_tasks=num_tasks, per_task_rotation=per_task_rotation)
    test = RotatedMNISTDataset('./data/', train=False, download=True, transform=composed_transform, num_tasks=num_tasks, per_task_rotation=per_task_rotation)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, generator=g)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=g)

    return train_loader, test_loader


def tasks_rotMNIST(num_tasks, per_task_rotation, batch_size, transform=[]):
    train_loaders = []
    test_loaders = []

    g = torch.Generator()
    g.manual_seed(0)

    for task in range(num_tasks):
        rotation_degree = task * per_task_rotation
        extended_transform = transform.copy()
        extended_transform.extend([
            RotationTransform(rotation_degree),
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
        composed_transform = T.Compose(extended_transform)

        train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=composed_transform)
        test = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=composed_transform)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, generator=g)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=g)

        train_loaders.append({'loader': train_loader, 'task': task, 'rot': rotation_degree})
        test_loaders.append({'loader': test_loader, 'task': task, 'rot': rotation_degree})

    return train_loaders, test_loaders


def tasks_rotMNIST_datasets(num_tasks, per_task_rotation, transform=[]):
    train_datasets = []
    test_datasets = []

    for task in range(num_tasks):
        rotation_degree = task * per_task_rotation
        extended_transform = transform.copy()
        extended_transform.extend([
            RotationTransform(rotation_degree),
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
        composed_transform = T.Compose(extended_transform)

        train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=composed_transform)
        test = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=composed_transform)

        train_datasets.append(train)
        test_datasets.append(test)

    return train_datasets, test_datasets


def tasks_rotMNIST_custom_rotations(rotation_list, transform=[]):
    train_datasets = []
    test_datasets = []

    for angle in rotation_list:
        extended_transform = transform.copy()
        extended_transform.extend([
            RotationTransform(angle),
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
        composed_transform = T.Compose(extended_transform)

        train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=composed_transform)
        test = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=composed_transform)

        train_datasets.append(train)
        test_datasets.append(test)

    return train_datasets, test_datasets


def generate_random_rotation_list(num_tasks, seed):
    np.random.seed(seed)
    return sorted(np.random.choice(range(360), size=num_tasks, replace=False).tolist())
