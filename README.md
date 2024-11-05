<img src="./rf.png" width="400px"></img>

## Rectified Flow - Pytorch

Implementation of <a href="https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html">rectified flow</a> and some of its followup research / improvements in Pytorch

<img src="./images/oxford-flowers.sample.png" width="350px"></img>

*32 batch size, 11k steps oxford flowers*

## Install

```bash
$ pip install rectified-flow-pytorch
```

## Usage

```python
import torch
from rectified_flow_pytorch import RectifiedFlow, Unet

model = Unet(dim = 64)

rectified_flow = RectifiedFlow(model)

images = torch.randn(1, 3, 256, 256)

loss = rectified_flow(images)
loss.backward()

sampled = rectified_flow.sample()
assert sampled.shape[1:] == images.shape[1:]
```

For reflow as described in the paper

```python
import torch
from rectified_flow_pytorch import RectifiedFlow, Reflow, Unet

model = Unet(dim = 64)

rectified_flow = RectifiedFlow(model)

images = torch.randn(1, 3, 256, 256)

loss = rectified_flow(images)
loss.backward()

# do the above for many real images

reflow = Reflow(rectified_flow)

reflow_loss = reflow()
reflow_loss.backward()

# then do the above in a loop many times for reflow - you can reflow multiple times by redefining Reflow(reflow.model) and looping again

sampled = reflow.sample()
assert sampled.shape[1:] == images.shape[1:]
```

With a `Trainer` based on `accelerate`

```python
import torch
from rectified_flow_pytorch import RectifiedFlow, ImageDataset, Unet, Trainer

model = Unet(dim = 64)

rectified_flow = RectifiedFlow(model)

img_dataset = ImageDataset(
    folder = './path/to/your/images',
    image_size = 256
)

trainer = Trainer(
    rectified_flow,
    dataset = img_dataset,
    num_train_steps = 70_000,
    results_folder = './results'   # samples will be saved periodically to this folder
)

trainer()
```

## Citations

```bibtex
@article{Liu2022FlowSA,
    title   = {Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
    author  = {Xingchao Liu and Chengyue Gong and Qiang Liu},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2209.03003},
    url     = {https://api.semanticscholar.org/CorpusID:252111177}
}
```

```bibtex
@article{Lee2024ImprovingTT,
    title   = {Improving the Training of Rectified Flows},
    author  = {Sangyun Lee and Zinan Lin and Giulia Fanti},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2405.20320},
    url     = {https://api.semanticscholar.org/CorpusID:270123378}
}
```

```bibtex
@article{Esser2024ScalingRF,
    title   = {Scaling Rectified Flow Transformers for High-Resolution Image Synthesis},
    author  = {Patrick Esser and Sumith Kulal and A. Blattmann and Rahim Entezari and Jonas Muller and Harry Saini and Yam Levi and Dominik Lorenz and Axel Sauer and Frederic Boesel and Dustin Podell and Tim Dockhorn and Zion English and Kyle Lacey and Alex Goodwin and Yannik Marek and Robin Rombach},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2403.03206},
    url     = {https://api.semanticscholar.org/CorpusID:268247980}
}
```

```bibtex
@article{Li2024ImmiscibleDA,
    title   = {Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment},
    author  = {Yiheng Li and Heyang Jiang and Akio Kodaira and Masayoshi Tomizuka and Kurt Keutzer and Chenfeng Xu},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.12303},
    url     = {https://api.semanticscholar.org/CorpusID:270562607}
}
```

```bibtex
@article{Yang2024ConsistencyFM,
    title   = {Consistency Flow Matching: Defining Straight Flows with Velocity Consistency},
    author  = {Ling Yang and Zixiang Zhang and Zhilong Zhang and Xingchao Liu and Minkai Xu and Wentao Zhang and Chenlin Meng and Stefano Ermon and Bin Cui},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2407.02398},
    url     = {https://api.semanticscholar.org/CorpusID:270878436}
}
```

```bibtex
@inproceedings{Yao2024FasterDiTTF,
    title   = {FasterDiT: Towards Faster Diffusion Transformers Training without Architecture Modification},
    author  = {Jingfeng Yao and Wang Cheng and Wenyu Liu and Xinggang Wang},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273346237}
}
```
