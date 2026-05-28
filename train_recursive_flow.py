# /// script
# dependencies = [
#   "accelerate",
#   "datasets",
#   "fire",
#   "rectified-flow-pytorch>=0.10.1",
#   "torchvision",
# ]
# ///

import fire
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision.utils import save_image

from rectified_flow_pytorch.rectified_flow import Unet, Trainer
from rectified_flow_pytorch.recursive_flow import RecursiveFlow

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

def main(
    image_size = 64,
    dim = 64,
    num_train_steps = 70_000,
    learning_rate = 1e-4,
    results_folder = './results_recursive',
    batch_size = 16
):
    flowers_dataset = OxfordFlowersDataset(
        image_size = image_size
    )

    model = Unet(
        dim = dim,
        accept_cond = True,
        dim_cond = 1,
        accept_dest_time = False
    )

    recursive_flow = RecursiveFlow(
        model,
        normalize_data_fn = lambda t: t * 2. - 1.,
        unnormalize_data_fn = lambda t: (t + 1.) / 2.,
        times_cond_kwarg = 'times',
        alphas_cond_kwargs = 'cond',
        predict_clean = False
    )

    trainer = Trainer(
        recursive_flow,
        dataset = flowers_dataset,
        num_train_steps = num_train_steps,
        learning_rate = learning_rate,
        batch_size = batch_size,
        results_folder = results_folder,
        adam_kwargs={'fused': True} if torch.cuda.is_available() else {},
    )

    trainer()

if __name__ == '__main__':
    fire.Fire(main)
