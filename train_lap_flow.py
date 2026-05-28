import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset

from rectified_flow_pytorch import LapFlow, LapFlowDiT, Trainer


class OxfordFlowersDataset(Dataset):
    def __init__(self, image_size):
        self.ds = load_dataset('nelorth/oxford-flowers')['train']

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        pil = item['image']
        label = item['label']

        tensor = self.transform(pil)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return tensor / 255., label_tensor


IMG_SIZE = 64
BATCH_SIZE = 8
CHANNELS = 3
NUM_SCALES = 2

is_cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda_available else 'cpu')

dataset = OxfordFlowersDataset(image_size=IMG_SIZE)

model = LapFlowDiT(
    base_image_size=IMG_SIZE,
    patch_size=2,
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=1024,
    channels=CHANNELS,
    num_scales=NUM_SCALES,
    accept_cond=True,
    dim_cond=1
)

lap_flow = LapFlow(
    model=model,
    num_scales=NUM_SCALES,
    normalize_data_fn=lambda t: (t * 2) - 1,
    unnormalize_data_fn=lambda t: (t + 1) * 0.5,
    cfg_scale=3.0
)


if __name__ == '__main__':

    trainer = Trainer(
        lap_flow,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        num_train_steps=100000,
        save_results_every=1000,
        checkpoint_every=5000,
        grad_accum_every = 4,
        use_ema=True,
        ema_kwargs={'beta': 0.9999}
    )

    trainer()
