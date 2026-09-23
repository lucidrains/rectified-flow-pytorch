import fire
import torch

# hf datasets for easy oxford flowers training

import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset

class OxfordFlowersDataset(Dataset):
    def __init__(
        self,
        image_size
    ):
        self.ds = load_dataset('nelorth/oxford-flowers')['train']

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil = self.ds[idx]['image']
        tensor = self.transform(pil)
        return tensor / 255.

# models and trainer

from rectified_flow_pytorch import RectifiedFlow, Unet, Trainer

def train(
    image_size = 64,
    dim = 64,
    batch_size = 16,
    num_train_steps = 70_000,
    sample_temperature = 1.5,
    save_results_every = 100,
    results_folder = './results',
    clear_results_folder = True
):
    flowers_dataset = OxfordFlowersDataset(
        image_size = image_size
    )

    model = Unet(
        dim = dim,
        mean_variance_net = False
    )

    rectified_flow = RectifiedFlow(model, predict = 'clean')

    trainer = Trainer(
        rectified_flow,
        dataset = flowers_dataset,
        batch_size = batch_size,
        num_train_steps = num_train_steps,
        sample_temperature = sample_temperature,
        save_results_every = save_results_every,
        results_folder = results_folder,
        clear_results_folder = clear_results_folder
    )

    trainer()

if __name__ == '__main__':
    fire.Fire(train)
