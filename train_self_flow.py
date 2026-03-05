# /// script
# dependencies = [
#   "accelerate",
#   "datasets",
#   "rectified-flow-pytorch>=0.8.0",
#   "torchvision",
# ]
# ///

import fire
import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset

from rectified_flow_pytorch.fit import FiT
from rectified_flow_pytorch.self_flow import SelfFlow
from rectified_flow_pytorch import Trainer

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
        return tensor.float() / 255.

def train(
    dim = 256,
    patch_size = 4,
    depth = 6,
    heads = 8,
    dim_head = 64,
    num_train_steps = 50_000,
    learning_rate = 1e-4,
    batch_size = 32,
    image_size = 64,
    save_results_every = 100,
    checkpoint_every = 500,
    clear_results_folder = True,
    checkpoints_folder = './checkpoints_self_flow',
    results_folder = './results_self_flow',
    predict_clean = False
):
    model = FiT(
        dim = dim,
        patch_size = patch_size,
        depth = depth,
        heads = heads,
        dim_head = dim_head
    )

    self_flow = SelfFlow(
        model = model,
        predict_clean = predict_clean
    )

    flowers_dataset = OxfordFlowersDataset(
        image_size = image_size
    )

    trainer = Trainer(
        self_flow,
        dataset = flowers_dataset,
        num_train_steps = num_train_steps,
        learning_rate = learning_rate,
        batch_size = batch_size,
        checkpoints_folder = checkpoints_folder,
        results_folder = results_folder,
        clear_results_folder = clear_results_folder,
        save_results_every = save_results_every,
        checkpoint_every = checkpoint_every,
    )

    trainer()

if __name__ == '__main__':
    fire.Fire(train)
