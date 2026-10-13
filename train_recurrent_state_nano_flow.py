import fire
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as T

from rectified_flow_pytorch import RecurrentStateNanoFlow, Unet, Trainer

# datasets

class MnistDataset(Dataset):
    def __init__(self, root = './data', download = True):
        self.ds = datasets.MNIST(root, train = True, download = download)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil, _ = self.ds[idx]
        return T.ToTensor()(pil)

class OxfordFlowersDataset(Dataset):
    def __init__(self, image_size = 64):
        from datasets import load_dataset

        self.ds = load_dataset('nelorth/oxford-flowers')['train']
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.transform(self.ds[idx]['image']).float() / 255.

# training
#
# ablate = False : train with the recurrent state pathway (triadic by default)
# ablate = True  : train the same unet as a plain flow model (memory dropped)
#
#   python train_recurrent_state_nano_flow.py --dataset flowers
#   python train_recurrent_state_nano_flow.py --dataset flowers --ablate
#   python train_recurrent_state_nano_flow.py --dataset mnist

def train(
    dataset = 'flowers',
    ablate = False,
    batch_size = 8,
    grad_accum_every = 2,
    num_train_steps = 20_000,
    recurrence_steps = 2,
    key_dims = (16, 4),
    learning_rate = 3e-4,
    dim = 32,
    num_heads = 4,
    save_results_every = 250,
    checkpoint_every = 1000,
    results_folder = None,
    checkpoints_folder = None
):
    if dataset == 'mnist':
        image_size, channels, dim_mults = 28, 1, (1, 2, 4)
        ds = MnistDataset()
    elif dataset == 'flowers':
        image_size, channels, dim_mults = 64, 3, (1, 2, 4, 8)
        ds = OxfordFlowersDataset(image_size = image_size)
    else:
        raise ValueError(f'unknown dataset {dataset}')

    mode_name = 'ablate' if ablate else f'triadic-fbt{recurrence_steps}'
    tag = f'{dataset}-{mode_name}'
    results_folder = results_folder or f'./results_recurrent_state_nano_flow/{tag}'
    checkpoints_folder = checkpoints_folder or f'./checkpoints_recurrent_state_nano_flow/{tag}'

    os.makedirs(results_folder, exist_ok = True)
    os.makedirs(checkpoints_folder, exist_ok = True)

    model = Unet(
        dim = dim,
        channels = channels,
        dim_mults = dim_mults
    )

    flow = RecurrentStateNanoFlow(
        model,
        num_heads = num_heads,
        key_dims = key_dims,
        num_train_steps = recurrence_steps,
        ablate = ablate,
        data_shape = (channels, image_size, image_size),
        normalize_data_fn = lambda t: t * 2. - 1.,
        unnormalize_data_fn = lambda t: (t + 1.) / 2.
    )

    trainer = Trainer(
        flow,
        dataset = ds,
        batch_size = batch_size,
        grad_accum_every = grad_accum_every,
        num_train_steps = num_train_steps,
        learning_rate = learning_rate,
        save_results_every = save_results_every,
        checkpoint_every = checkpoint_every,
        results_folder = results_folder,
        checkpoints_folder = checkpoints_folder,
        clear_results_folder = False
    )

    print(f'Training RecurrentStateNanoFlow [{tag}] on {dataset} ({image_size}x{image_size})')
    print(f'ablate: {ablate} | key_dims: {key_dims} | recurrence_steps: {recurrence_steps}')
    print(f'results: {results_folder} | checkpoints: {checkpoints_folder}')

    trainer()

if __name__ == '__main__':
    fire.Fire(train)
