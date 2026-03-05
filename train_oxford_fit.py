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

from rectified_flow_pytorch import RectifiedFlow, FiT, Trainer

model = FiT(
    dim = 256,
    patch_size = 4,
    channels = 3,
    depth = 8,
    heads = 8,
    dim_head = 64
)

rectified_flow = RectifiedFlow(model, predict = 'clean')

trainer = Trainer(
    rectified_flow,
    dataset = flowers_dataset,
    num_train_steps = 70_000,
    sample_temperature = 1.0,
    results_folder = './results_fit'   # samples will be saved periodically to this folder
)

trainer()
