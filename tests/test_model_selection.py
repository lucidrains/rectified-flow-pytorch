import torch
from torch import nn

from rectified_flow_pytorch.rectified_flow import RectifiedFlow


class ConstantFlow(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.value = value
        self.calls = 0

    def forward(self, noised, *, times):
        self.calls += 1
        return torch.full_like(noised, self.value)


def test_predict_flow_uses_the_requested_model():
    online_model = ConstantFlow(1.)
    selected_model = ConstantFlow(7.)
    flow = RectifiedFlow(online_model)

    noised = torch.zeros((2, 3, 4, 4))
    output, predicted_flow = flow.predict_flow(
        selected_model,
        noised,
        times = torch.tensor(0.5)
    )

    expected = torch.full_like(noised, 7.)
    assert torch.equal(output, expected)
    assert torch.equal(predicted_flow, expected)
    assert selected_model.calls == 1
    assert online_model.calls == 0
