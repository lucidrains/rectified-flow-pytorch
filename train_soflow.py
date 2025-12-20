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

from rectified_flow_pytorch.rectified_flow import Unet
from rectified_flow_pytorch.soflow import SoFlow, SoFlowTrainer

# constants

is_cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda_available else 'cpu')

model = Unet(
    dim = 64,
    accept_cond = False
)

soflow = SoFlow(
    model,
    lambda_flow_matching = 0.75,
    model_output_clean = True,
    normalize_data_fn = lambda t: t * 2. - 1.,
    unnormalize_data_fn = lambda t: (t + 1.) / 2.,
    use_adaptive_loss_weight = True,
    adaptive_loss_weight_p = 0.5,
    r_schedule = 'exponential',
    r_init = 0.1,
    r_end = 0.002
).to(device)

num_params = sum(p.numel() for p in soflow.model.parameters() if p.requires_grad)
print(f"Trainable parameters in the model: {num_params:,}")

trainer = SoFlowTrainer(
    soflow,
    dataset = flowers_dataset,
    num_train_steps = 70_000,
    learning_rate = 1e-4,
    results_folder = './results',
    adam_kwargs={'fused': True},
    dl_kwargs={
        'prefetch_factor': 2 if is_cuda_available else None,
        'persistent_workers': True if is_cuda_available else False,
        'num_workers': 4 if is_cuda_available else 0,
        'pin_memory': True if is_cuda_available else False
    }
)

trainer()
