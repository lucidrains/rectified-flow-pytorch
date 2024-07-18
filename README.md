## Rectified Flow - Pytorch (wip)

Implementation of rectified flow and some of its followup research / improvements in Pytorch

## Install

```bash
$ pip install rectified-flow-pytorch
```

## Usage

```python
import torch
from torch import nn

from rectified_flow_pytorch import RectifiedFlow

model = nn.Conv2d(3, 3, 1)

rectified_flow = RectifiedFlow(model, time_cond_kwarg = None)

images = torch.randn(1, 3, 256, 256)

loss = rectified_flow(images)
loss.backward()

sampled = rectified_flow.sample()
assert sampled.shape[1:] == images.shape[1:]
```

For reflow as described in the paper

```python
import torch
from torch import nn

from rectified_flow_pytorch import RectifiedFlow, Reflow

model = nn.Conv2d(3, 3, 1)

rectified_flow = RectifiedFlow(model, time_cond_kwarg = None)

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
