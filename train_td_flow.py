# /// script
# dependencies = [
#   "rectified-flow-pytorch>=0.7.1",
#   "torch",
#   "torchvision",
#   "fire",
#   "tqdm",
#   "numpy",
#   "einops",
#   "accelerate",
# ]
# ///

from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid

import fire
from tqdm import tqdm
from einops import repeat
from accelerate import Accelerator

from rectified_flow_pytorch.rectified_flow import Unet
from rectified_flow_pytorch.td_flow import TDFlow

# dataset

class TDMovingMNISTDataset(Dataset):
    def __init__(
        self,
        root = './data',
        image_size = 32,
        digit_size = 14,
        min_velocity = -2,
        max_velocity = 3,
        download = True
    ):
        super().__init__()
        from torchvision.datasets import MNIST
        self.mnist = MNIST(root = root, train = True, download = download)
        self.image_size = image_size
        self.digit_size = digit_size
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        digit, _ = self.mnist[idx]
        digit = T.functional.resize(digit, (self.digit_size, self.digit_size))
        digit = T.functional.to_tensor(digit)[0]  # (digit_size, digit_size)

        x = np.random.randint(0, self.image_size - self.digit_size)
        y = np.random.randint(0, self.image_size - self.digit_size)
        vx = np.random.randint(self.min_velocity, self.max_velocity)
        vy = np.random.randint(self.min_velocity, self.max_velocity)

        def place_digit(px, py):
            frame = torch.zeros(self.image_size, self.image_size)
            px = int(np.clip(px, 0, self.image_size - self.digit_size))
            py = int(np.clip(py, 0, self.image_size - self.digit_size))
            frame[py:py + self.digit_size, px:px + self.digit_size] = digit
            return repeat(frame, 'h w -> c h w', c = 3)

        state = place_digit(x, y)
        next_state = place_digit(x + vx, y + vy)

        return state, next_state

# helpers

def divisible_by(num, den):
    return (num % den) == 0

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# main

def main(
    num_train_steps = 20_000,
    batch_size = 16,
    lr = 3e-4,
    dim = 64,
    num_samples = 4,
    num_sample_rows = 4,
    save_results_every = 100,
    results_folder = './results_td_flow',
    checkpoints_folder = './checkpoints_td_flow',
    image_size = 32,
    horizon_consistency = True,
    condition_on_discount = True
):
    accelerator = Accelerator()
    device = accelerator.device

    results_folder = Path(results_folder)
    checkpoints_folder = Path(checkpoints_folder)

    if results_folder.exists():
        rmtree(str(results_folder))

    results_folder.mkdir(exist_ok = True, parents = True)
    checkpoints_folder.mkdir(exist_ok = True, parents = True)

    # dataset

    dataset = TDMovingMNISTDataset(image_size = image_size)
    dl = cycle(DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True))

    # model

    model = Unet(
        dim = dim,
        dim_mults = (1, 2, 4),
        channels = 3,
        has_image_cond = True,
        accept_cond = condition_on_discount,
        dim_cond = 3 if condition_on_discount else None
    )

    td_flow = TDFlow(
        model,
        horizon_consistency = horizon_consistency,
        condition_on_discount = condition_on_discount,
        flow_kwargs = dict(
            normalize_data_fn = lambda t: t * 2. - 1.,
            unnormalize_data_fn = lambda t: (t + 1.) / 2.
        )
    )

    optimizer = torch.optim.Adam(td_flow.parameters(), lr = lr)

    td_flow, optimizer = accelerator.prepare(td_flow, optimizer)

    # training

    pbar = tqdm(range(1, num_train_steps + 1))

    for step in pbar:
        td_flow.train()

        state, next_state = next(dl)
        state, next_state = state.to(device), next_state.to(device)

        loss = td_flow(state, next_state)

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(td_flow.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        accelerator.unwrap_model(td_flow).update_ema()

        pbar.set_description(f'loss: {loss.item():.4f}')

        # sample

        if divisible_by(step, save_results_every):
            td_flow.eval()

            with torch.no_grad():
                prompt_state, _ = next(dl)
                prompt_state = prompt_state[:num_sample_rows].to(device)

                # for each prompt row: [prompt | sample_1 | sample_2 | ... | sample_n]

                images = []

                for prompt in prompt_state:
                    images.append(prompt.cpu())
                    prompt_batch = prompt.unsqueeze(0)

                    for _ in range(num_samples):
                        sample = accelerator.unwrap_model(td_flow)(prompt_batch, is_training = False)
                        images.append(sample.squeeze(0).clamp(0., 1.).cpu())

                grid = make_grid(images, nrow = num_samples + 1, padding = 2, pad_value = 1.)
                save_image(grid, str(results_folder / f'step_{step}.png'))
                print(f'\nsamples saved to {results_folder}/step_{step}.png')

if __name__ == '__main__':
    fire.Fire(main)
