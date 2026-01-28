from torch.optim import Adam, SGD
from torch.utils.data import Dataset
import torchvision.transforms as T
from datasets import load_dataset
from rectified_flow_pytorch import EquilibriumMatching, Unet, Trainer

# hf datasets for easy oxford flowers training

class OxfordFlowersDataset(Dataset):
    def __init__(
        self,
        image_size
    ):
        self.ds = load_dataset('nelorth/oxford-flowers')['train']

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil = self.ds[idx]['image']
        tensor = self.transform(pil)
        return tensor

# dataset

flowers_dataset = OxfordFlowersDataset(
    image_size = 64
)

# model - now using unet without time conditioning!

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    num_residual_streams = 4,
    accept_time = False
)

# equilibrium matching

eqm = EquilibriumMatching(
    model,
    decay_kwargs = dict(a = 0.8),
    lambda_multiplier = 4.0,
    sample_optim = Adam,
    sample_optim_kwargs = dict(lr = 0.003)
)

# trainer

trainer = Trainer(
    eqm,
    dataset = flowers_dataset,
    num_train_steps = 10_000,
    batch_size = 16,
    save_results_every = 100,
    checkpoint_every = 500,
    results_folder = './results',
    clear_results_folder = True
)

trainer()
