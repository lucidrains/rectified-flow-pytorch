import fire
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from datasets import load_dataset

from rectified_flow_pytorch import ImageDataset, Unet
from rectified_flow_pytorch.mgm import MGM, MGMTrainer
from rectified_flow_pytorch.rectified_flow import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one
)

class OxfordFlowersDataset(Dataset):
    def __init__(self, image_size = 64, augment_horizontal_flip = True):
        flowers = load_dataset('nelorth/oxford-flowers')['train']

        transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.PILToTensor()
        ])

        self.images = torch.stack([transform(row['image']) for row in flowers])
        self.flip = T.RandomHorizontalFlip() if augment_horizontal_flip else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].float() / 255.
        return self.flip(image) if self.flip is not None else image

def train(
    folder = None,
    image_size = 64,
    dim = 32,
    dim_mults = (1, 2, 4),
    batch_size = 16,
    warmup_steps = 400,
    critic_warmup_steps = 20,
    num_train_steps = 1000,
    learning_rate = 5e-5,
    critic_learning_rate = 1e-4,
    critic_steps = 1,
    sigma = 0.2,
    critic_only_noise = True,
    grad_penalty_weight = 1.0,
    grad_penalty_every = 4,
    ema_decay = 0.99,
    save_results_every = 100,
    results_folder = './results_mgm',
    checkpoints_folder = './checkpoints_mgm',
    clear_results_folder = True
):
    dataset = (
        ImageDataset(folder = folder, image_size = image_size, augment_horizontal_flip = True)
        if folder is not None else
        OxfordFlowersDataset(image_size = image_size)
    )

    mgm = MGM(
        generator = Unet(dim = dim, dim_mults = dim_mults),
        data_shape = (3, image_size, image_size),
        sigma = sigma,
        critic_only_noise = critic_only_noise,
        grad_penalty_weight = grad_penalty_weight,
        normalize_data_fn = normalize_to_neg_one_to_one,
        unnormalize_data_fn = unnormalize_to_zero_to_one
    )

    trainer = MGMTrainer(
        mgm,
        dataset = dataset,
        num_train_steps = num_train_steps,
        learning_rate = learning_rate,
        critic_learning_rate = critic_learning_rate,
        critic_steps = critic_steps,
        warmup_steps = warmup_steps,
        critic_warmup_steps = critic_warmup_steps,
        grad_penalty_every = grad_penalty_every,
        batch_size = batch_size,
        ema_kwargs = dict(beta = ema_decay, update_after_step = 10),
        save_results_every = save_results_every,
        results_folder = results_folder,
        checkpoints_folder = checkpoints_folder,
        clear_results_folder = clear_results_folder
    )

    trainer()

if __name__ == '__main__':
    fire.Fire(train)
