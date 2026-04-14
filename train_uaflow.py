import torch
from torch.nn import Module
import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset

from rectified_flow_pytorch import UAFlow, Trainer, Unet

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
BATCH_SIZE = 32
CHANNELS = 3

is_cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda_available else 'cpu')

dataset = OxfordFlowersDataset(image_size=IMG_SIZE)

unet = Unet(
    dim=64,
    channels=CHANNELS, 
    accept_cond=True,
    dim_cond=1,
    mean_variance_net=True
    )


ua_flow = UAFlow(
    model=unet, 
    times_cond_kwarg='times',
    ucg_scale=2,
    cfg_scale=3,
    normalize_data_fn = lambda t: t * 2. - 1.,
    unnormalize_data_fn = lambda t: (t + 1.) / 2.,
)


if __name__ == '__main__':

    trainer = Trainer(
        ua_flow,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        num_train_steps=1000000,
        save_results_every=5000,
        checkpoint_every=50000,   
    )

    trainer()