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

class TestProtection(unittest.TestCase):
    def test_pgd_cpu(self):
        # Setup
        device = "cpu"
        dtype = torch.float32

        # Dummy image (Batch, Channel, Height, Width)
        # Latents are usually smaller.
        # utils.pgd takes X (latents or image?)
        # In app.py: X_full = utils.preprocess(init_image) -> Image space (1, 3, 512, 512)
        # And model is pipe.vae.encode.
        # pipe.vae.encode takes Image Space, returns Latents.

        X = torch.randn(1, 3, 512, 512, device=device, dtype=dtype)
        targets = torch.randn(1, 4, 64, 64, device=device, dtype=dtype) # Latent shape

        # Mock Model
        mock_model = MagicMock()
        # When model(input) is called, it should return an output with .latent_dist.mean
        # The output mean should be differentiable w.r.t input.
        # So we can't just return a constant. We need a simple function.

        # Define a simple function that simulates VAE encode
        # Input: (1, 3, 512, 512) -> Output Latent: (1, 4, 64, 64)
        # Just do a conv or adaptive avg pool to simulate
        conv = torch.nn.Conv2d(3, 4, kernel_size=8, stride=8, padding=0).to(dtype)

        def model_forward(x):
            out = conv(x) # (1, 4, 64, 64)
            return MockModelOutput(out)

        mock_model.side_effect = model_forward

        # Run PGD
        iters = 5
        eps = 0.1
        X_adv = utils.pgd(X,
                          model=mock_model,
                          targets=targets,
                          iters=iters,
                          eps=eps,
                          step_size=0.01)

        # Check output shape
        self.assertEqual(X_adv.shape, X.shape)

        # Check that X_adv is different from X
        self.assertFalse(torch.allclose(X_adv, X))

        # Check constraints (L_inf norm <= eps)
        # Note: We must compare against the ORIGINAL X passed to the function.
        # But wait, X_adv is returned.
        # X is modified in place? No, X.clone().detach().

        # Why is the diff large?
        # Maybe because X is random noise and clamp happens later?
        # Ah, X in test is randn, so values can be large (outside -1, 1).
        # utils.pgd clamps to [clamp_min, clamp_max] (-1, 1) or (0, 1) at the END of step.
        # If X was initially OUTSIDE [clamp_min, clamp_max], and X_adv is clamped,
        # then X_adv - X could be large regardless of eps constraint on delta!

        # PGD assumes X is valid (within min/max).
        # Let's ensure X is valid in test.
        # Note: pgd defaults to clamp_min=0, clamp_max=1 in my implementation?
        # Let's check utils_protection.py signature:
        # def pgd(..., clamp_min=0, clamp_max=1, ...)
        # So if X is in [-1, 1] but pgd clamps to [0, 1], then X_adv will be far from X!
        # X should be in range [clamp_min, clamp_max].

        # Test Case adjustment: use clamp_min=-1, clamp_max=1 to match preprocess output.
        iters = 5
        eps = 0.1
        # Re-run with correct clamps
        X = torch.clamp(X, -1, 1)
        X_adv = utils.pgd(X,
                          model=mock_model,
                          targets=targets,
                          iters=iters,
                          eps=eps,
                          step_size=0.01,
                          clamp_min=-1,
                          clamp_max=1)

        diff = X_adv - X
        print(f"Max Diff: {diff.abs().max()}, Eps: {eps}")

        # Also need to account for the clamp_min/max logic.
        # The constraint is |X_adv - X| <= eps AND min <= X_adv <= max.
        # If X is at boundary, X_adv stays at boundary?

        self.assertTrue(diff.abs().max() <= eps + 1e-5)

if __name__ == '__main__':
    unittest.main()
