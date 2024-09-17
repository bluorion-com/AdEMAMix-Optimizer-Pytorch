import torch
from absl.testing import absltest, parameterized

from third_party.ademamix.AdEMAMix import AdEMAMix


def _generate_data(A, batch_size, cuda_is_available):
    in_features = A.shape[1]
    while True:
        x = torch.randn(
            batch_size,
            in_features,
            device="cuda" if cuda_is_available else "cpu",
        )
        y = x @ A.T
        yield x, y


class AdEMAMixTest(parameterized.TestCase):
    def setUp(self):
        self.cuda_is_available = torch.cuda.is_available()

    @parameterized.parameters(
        ("AdamW", 5000),
        ("AdEMAMix", 10000),
    )
    def test_simple_linear(self, optimizer, steps):
        A = torch.randn(
            5,
            5,
            device="cuda" if self.cuda_is_available else "cpu",
        )
        data = _generate_data(A, 10000000, self.cuda_is_available)
        m = torch.nn.Linear(
            5,
            5,
            bias=False,
            device="cuda" if self.cuda_is_available else "cpu",
        )
        o = (
            torch.optim.AdamW(m.parameters(), lr=1e-2)
            if optimizer == "AdamW"
            else AdEMAMix(m.parameters(), lr=1e-2)
        )

        for _ in range(steps):
            x, y = next(data)
            o.zero_grad()
            y_pred = m(x)
            loss = torch.nn.functional.mse_loss(y_pred, y)
            loss.backward()
            o.step()

        torch.testing.assert_close(
            m.weight,
            A,
            atol=1e-3,
            rtol=1e-3,
        )


if __name__ == "__main__":
    absltest.main()
