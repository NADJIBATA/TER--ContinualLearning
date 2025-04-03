import copy
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset
from data.manipulate import permutate_image_pixels, SubDataset, TransformedDataset
from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS
from data.rotated_mnist import tasks_rotMNIST_datasets, tasks_rotMNIST_custom_rotations, generate_random_rotation_list , generate_random_rotation_list_random_rank

def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./store/datasets',
                verbose=False, augment=False, normalize=False, target_transform=None):
    data_name = 'MNIST' if name in ('MNIST28', 'MNIST32') else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name + "_norm"]]
    if permutation is not None:
        transforms_list.append(transforms.Lambda(lambda x, p=permutation: permutate_image_pixels(x, p)))
    dataset_transform = transforms.Compose(transforms_list)

    dataset = dataset_class(f'{dir}/{data_name}', train=(type != 'test'),
                            download=download, transform=dataset_transform, target_transform=target_transform)

    if verbose:
        print(f" --> {name}: '{type}'-dataset consisting of {len(dataset)} samples")

    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset

def get_singlecontext_datasets(name, data_dir="./store/datasets", normalize=False, augment=False, verbose=False):
    config = DATASET_CONFIGS[name]
    config['output_units'] = config['classes']
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[name + "_denorm"]

    trainset = get_dataset(name, type='train', dir=data_dir, verbose=verbose, normalize=normalize, augment=augment)
    testset = get_dataset(name, type='test', dir=data_dir, verbose=verbose, normalize=normalize)

    return (trainset, testset), config

def get_context_set(name, scenario, contexts, random_seed=None ,random_rank=None, distance=None, data_dir="./datasets", only_config=False,
                    verbose=False, exception=False, normalize=False, augment=False, singlehead=False, train_set_per_class=False):

    if name == "splitMNIST":
        data_type = 'MNIST'
    elif name == "permMNIST":
        data_type = 'MNIST32'
        if train_set_per_class:
            raise NotImplementedError('Permuted MNIST does not support separate training dataset per class')
    elif name == "CIFAR10":
        data_type = 'CIFAR10'
    elif name == "CIFAR100":
        data_type = 'CIFAR100'
    elif name == 'RotatedMNIST':
        data_type = 'RotatedMNIST'
    else:
        raise ValueError(f'Undefined experiment: {name}')

    config = DATASET_CONFIGS[data_type].copy()
    config['normalize'] = normalize if name == 'CIFAR100' else False
    if config['normalize']:
        config['denormalize'] = AVAILABLE_TRANSFORMS["CIFAR100_denorm"]

    if contexts > config['classes'] and not (name in ["permMNIST", "RotatedMNIST"]):
        raise ValueError(f"Experiment '{name}' cannot have more than {config['classes']} contexts!")

    classes_per_context = 10 if name in ["permMNIST", "RotatedMNIST"] else int(np.floor(config['classes'] / contexts))
    config['classes_per_context'] = classes_per_context
    config['output_units'] = classes_per_context if (scenario == 'domain' or (scenario == "task" and singlehead)) \
        else classes_per_context * contexts

    if only_config:
        return config

    if name == 'permMNIST':
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=None, verbose=verbose)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=None, verbose=verbose)

        permutations = [None] + [np.random.permutation(config['size'] ** 2) for _ in range(contexts - 1)] if exception \
            else [np.random.permutation(config['size'] ** 2) for _ in range(contexts)]

        train_datasets = []
        test_datasets = []
        for context_id, perm in enumerate(permutations):
            target_transform = transforms.Lambda(
                lambda y, x=context_id: y + x * classes_per_context
            ) if scenario in ('task', 'class') and not (scenario == 'task' and singlehead) else None

            train_datasets.append(TransformedDataset(
                trainset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
            test_datasets.append(TransformedDataset(
                testset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))

    elif name == "RotatedMNIST":
        if random_rank :
            rotations = generate_random_rotation_list_random_rank(contexts, seed=random_seed)
            print("\n[INFO] Rotations aléatoires générées:", rotations)
            train_datasets, test_datasets = tasks_rotMNIST_custom_rotations(rotations)

        elif random_seed is not None:
            rotations = generate_random_rotation_list(contexts, seed=random_seed)
            print("\n[INFO] Rotations aléatoires générées:", rotations)
            train_datasets, test_datasets = tasks_rotMNIST_custom_rotations(rotations)
        else:
            per_task_rotation = distance if distance is not None else 24
            train_datasets, test_datasets = tasks_rotMNIST_datasets(num_tasks=contexts, per_task_rotation=per_task_rotation)

    else:
        classes = config['classes']
        perm_class_list = np.array(list(range(classes))) if exception else np.random.permutation(list(range(classes)))
        target_transform = transforms.Lambda(lambda y, p=perm_class_list: int(p[y]))

        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=target_transform,
                               verbose=verbose, augment=augment, normalize=normalize)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=target_transform,
                              verbose=verbose, augment=augment, normalize=normalize)

        labels_per_dataset_train = [[label] for label in range(classes)] if train_set_per_class else [
            list(np.array(range(classes_per_context)) + classes_per_context * context_id)
            for context_id in range(contexts)
        ]
        labels_per_dataset_test = [
            list(np.array(range(classes_per_context)) + classes_per_context * context_id)
            for context_id in range(contexts)
        ]

        train_datasets = []
        for labels in labels_per_dataset_train:
            target_transform = transforms.Lambda(lambda y, x=labels[0]: y - x) if (
                    scenario == 'domain' or (scenario == 'task' and singlehead)) else None
            train_datasets.append(SubDataset(trainset, labels, target_transform=target_transform))

        test_datasets = []
        for labels in labels_per_dataset_test:
            target_transform = transforms.Lambda(lambda y, x=labels[0]: y - x) if (
                    scenario == 'domain' or (scenario == 'task' and singlehead)) else None
            test_datasets.append(SubDataset(testset, labels, target_transform=target_transform))

    print(config)
    return ((train_datasets, test_datasets), config)