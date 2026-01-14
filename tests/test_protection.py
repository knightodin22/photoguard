import torch
import unittest
from unittest.mock import MagicMock
import src.utils_protection as utils

class MockLatentDist:
    def __init__(self, mean):
        self.mean = mean

class MockModelOutput:
    def __init__(self, mean):
        self.latent_dist = MockLatentDist(mean)
        self.sample = mean # For decode simulation

class MockUNetOutput:
    def __init__(self, sample):
        self.sample = sample

class TestProtection(unittest.TestCase):
    def test_pgd_cpu(self):
        # Setup
        device = "cpu"
        dtype = torch.float32

        X = torch.randn(1, 3, 512, 512, device=device, dtype=dtype)
        targets = torch.randn(1, 4, 64, 64, device=device, dtype=dtype) # Latent shape

        # Mock Model
        mock_model = MagicMock()
        conv = torch.nn.Conv2d(3, 4, kernel_size=8, stride=8, padding=0).to(dtype)

        def model_forward(x):
            out = conv(x) # (1, 4, 64, 64)
            return MockModelOutput(out)

        mock_model.side_effect = model_forward

        # Run PGD
        iters = 5
        eps = 0.1
        X = torch.clamp(X, -1, 1)
        X_adv = utils.pgd(X,
                          model=mock_model,
                          targets=targets,
                          iters=iters,
                          eps=eps,
                          step_size=0.01,
                          clamp_min=-1,
                          clamp_max=1)

        # Check output shape
        self.assertEqual(X_adv.shape, X.shape)

        # Check constraints (L_inf norm <= eps)
        diff = X_adv - X
        # print(f"Max Diff: {diff.abs().max()}, Eps: {eps}")
        self.assertTrue(diff.abs().max() <= eps + 1e-5)

    def test_super_l2_mock(self):
        # Smoke test for super_l2 logic (Advanced Attack)
        # Using a custom class to mock pipeline structure differentiably

        class FakeTokenizer:
            model_max_length = 77
            def __call__(self, *args, **kwargs):
                return MagicMock(input_ids=torch.zeros((1, 77), dtype=torch.long))

        class FakeTextEncoder(torch.nn.Module):
            def forward(self, x):
                return [torch.randn(1, 77, 768)]

        class FakeDist:
            def __init__(self, sample_tensor):
                self._sample = sample_tensor
            def sample(self):
                return self._sample

        class FakeVAEEncOut:
            def __init__(self, dist):
                self.latent_dist = dist

        class FakeVAEDecOut:
            def __init__(self, sample):
                self.sample = sample

        class FakeVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MagicMock()
                self.config.latent_channels = 4

            def encode(self, x):
                # Differentiable downsample
                out = torch.nn.functional.interpolate(x, size=(64, 64))
                out = out.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return FakeVAEEncOut(FakeDist(out))

            def decode(self, latents):
                # Differentiable upsample
                out = torch.nn.functional.interpolate(latents, size=(512, 512))
                out = out[:, :3, :, :]
                return FakeVAEDecOut(out)

        class FakeScheduler:
            init_noise_sigma = 1.0
            timesteps = torch.tensor([1, 0])
            def set_timesteps(self, *args): pass
            def step(self, noise_pred, t, latents, **kwargs):
                # Return combination of latents and noise_pred to keep gradient flow
                mock_out = MagicMock()
                # Crucial: Must involve noise_pred for gradient to flow from Unet back to input
                mock_out.prev_sample = latents + noise_pred
                return mock_out

        class FakeUNet(torch.nn.Module):
            def forward(self, latent_model_input, t, encoder_hidden_states):
                # Return first 4 channels
                return MockUNetOutput(latent_model_input[:, :4, :, :])

        class FakePipe:
            def __init__(self):
                self.device = "cpu"
                self.tokenizer = FakeTokenizer()
                self.text_encoder = FakeTextEncoder()
                self.vae = FakeVAE()
                self.scheduler = FakeScheduler()
                self.unet = FakeUNet()

        pipe = FakePipe()

        # Inputs
        mask = torch.zeros(1, 1, 512, 512)
        X = torch.randn(1, 3, 512, 512)
        target = torch.randn(1, 3, 512, 512)

        # Run
        try:
            X_adv = utils.super_l2(
                pipe=pipe,
                cur_mask=mask,
                X=X,
                prompt="test",
                step_size=0.1,
                iters=1,
                eps=0.1,
                clamp_min=-1,
                clamp_max=1,
                grad_reps=1,
                target_image=target
            )
            success = True
        except Exception as e:
            print(f"super_l2 failed: {e}")
            success = False

        self.assertTrue(success)
        self.assertEqual(X_adv.shape, X.shape)

if __name__ == '__main__':
    unittest.main()
