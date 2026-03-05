# /// script
# dependencies = [
#   "rectified-flow-pytorch>=0.7.1",
#   "torch",
#   "torchvision",
#   "fire",
#   "tqdm",
#   "numpy",
#   "einops",
# ]
# ///

from math import sqrt
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image

import fire
from tqdm import tqdm
from einops import rearrange, repeat

from rectified_flow_pytorch.video_unet import VideoUnet
from rectified_flow_pytorch import RectifiedFlow, NanoFlow, Trainer

# dataset

class MovingMNISTDataset(Dataset):
    def __init__(
        self,
        root = './data',
        num_frames = 10,
        image_size = 64,
        digit_size = 28,
        min_velocity = -2,
        max_velocity = 3,
        download = True
    ):
        super().__init__()
        from torchvision.datasets import MNIST
        self.mnist = MNIST(root = root, train = True, download = download)
        self.num_frames = num_frames
        self.image_size = image_size
        self.digit_size = digit_size
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        # Create a simple moving MNIST digit
        digit, _ = self.mnist[idx]
        digit = T.functional.to_tensor(digit) # (1, 28, 28)

        # Random start position and velocity
        x = np.random.randint(0, self.image_size - self.digit_size)
        y = np.random.randint(0, self.image_size - self.digit_size)
        vx = np.random.randint(self.min_velocity, self.max_velocity)
        vy = np.random.randint(self.min_velocity, self.max_velocity)

        video = torch.zeros(self.num_frames, self.image_size, self.image_size)

        for f in range(self.num_frames):
            curr_x = int(np.clip(x + vx * f, 0, self.image_size - self.digit_size))
            curr_y = int(np.clip(y + vy * f, 0, self.image_size - self.digit_size))

            # Draw digit
            video[f, curr_y:curr_y+self.digit_size, curr_x:curr_x+self.digit_size] = torch.maximum(
                video[f, curr_y:curr_y+self.digit_size, curr_x:curr_x+self.digit_size],
                digit[0]
            )

        video = repeat(video, 'f h w -> 3 f h w')
        return video

# helper for video saving

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    # tensor: (C, F, H, W)
    images = [T.ToPILImage()(img) for img in tensor.unbind(dim = 1)]
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)

# sample saver module

class VideoSampleSaver(nn.Module):
    def __init__(self, duration = 120, loop = 0, optimize = True):
        super().__init__()
        self.duration = duration
        self.loop = loop
        self.optimize = optimize

    def forward(self, video, path):
        path = Path(path)
        gif_path = str(path.with_suffix('.gif'))

        batch = video.shape[0]
        num_rows = int(sqrt(batch))

        video = rearrange(video, '(row col) c f h w -> c f (row h) (col w)', row = num_rows)

        video_tensor_to_gif(
            video.cpu(),
            gif_path,
            duration = self.duration,
            loop = self.loop,
            optimize = self.optimize
        )

        print(f'sampled video saved to {gif_path}')

def main(
    num_frames = 10,
    num_train_steps = 20_000,
    batch_size = 4,
    grad_accum_every = 4,
    min_velocity = -2,
    max_velocity = 3,
    lr = 3e-4,
    dim = 32,
    num_samples = 4,
    save_results_every = 50,
    results_folder = './results_mnist',
    checkpoints_folder = './checkpoints_mnist',
    clear_results_folder = False
):
    # dataset

    dataset = MovingMNISTDataset(
        num_frames = num_frames,
        min_velocity = min_velocity,
        max_velocity = max_velocity
    )

    # model

    model = VideoUnet(
        dim = dim,
        dim_mults = (1, 2, 4)
    )

    # wrapper

    nano_flow = NanoFlow(
        model,
        times_cond_kwarg = 'times',
        predict_clean = True,
        normalize_data_fn = lambda t: t * 2. - 1.,
        unnormalize_data_fn = lambda t: (t + 1.) / 2.
    )

    # trainer

    trainer = Trainer(
        nano_flow,
        dataset = dataset,
        num_train_steps = num_train_steps,
        learning_rate = lr,
        batch_size = batch_size,
        grad_accum_every = grad_accum_every,
        num_samples = num_samples,
        save_results_every = save_results_every,
        results_folder = results_folder,
        checkpoints_folder = checkpoints_folder,
        clear_results_folder = clear_results_folder,
        save_sample_fn = VideoSampleSaver()
    )

    trainer()

if __name__ == '__main__':
    fire.Fire(main)
