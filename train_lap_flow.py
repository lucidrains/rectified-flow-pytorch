import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset
from diffusers import AutoencoderKL

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




use_vae = True

if use_vae:
    IMG_SIZE = 256
    kwargs = dict(
        base_image_size = IMG_SIZE // 8,
        channels = 4,
        num_scales = 2
    )
else:
    IMG_SIZE = 64
    kwargs = dict(
        base_image_size = 64,
        channels = 3,
        num_scales = 2
    )


is_cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda_available else 'cpu')

dataset = OxfordFlowersDataset(image_size=IMG_SIZE)

if use_vae:
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    for param in vae.parameters():
        param.requires_grad = False
else:
    vae = None

model = LapFlowDiT(
    **kwargs,
    patch_size=2,
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=1024,
    accept_cond=True,
    dim_cond=1
)

lap_flow = LapFlow(
    model=model,
    normalize_data_fn=lambda t: (t * 2) - 1,
    unnormalize_data_fn=lambda t: (t + 1) * 0.5,
    cfg_scale=3.0,
    vae=vae,
    vae_scale_factor=0.18215
)


if __name__ == '__main__':

    trainer = Trainer(
        lap_flow,
        dataset=dataset,
        batch_size=8,
        learning_rate=1e-4,
        num_train_steps=100000,
        save_results_every=1000,
        checkpoint_every=5000,
        grad_accum_every = 4,
        use_ema=True,
        ema_kwargs={'beta': 0.9999}
    )

    trainer()
