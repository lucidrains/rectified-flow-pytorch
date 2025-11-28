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

flowers_dataset = OxfordFlowersDataset(
    image_size = 64
)

# models and trainer

from rectified_flow_pytorch import Unet, Trainer
from rectified_flow_pytorch.nano_flow_multi_obj import NanoFlow

model = Unet(dim = 64, num_outputs = 3)

nano_flow = NanoFlow(
    model,
    times_cond_kwarg = 'times',
    normalize_data_fn = lambda t: t * 2. - 1.,
    unnormalize_data_fn = lambda t: (t + 1.) / 2.
)

trainer = Trainer(
    nano_flow,
    dataset = flowers_dataset,
    num_train_steps = 70_000,
    results_folder = './results'   # samples will be saved periodically to this folder
)

trainer()
