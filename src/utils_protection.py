import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage, ToTensor

totensor = ToTensor()
topil = ToPILImage()

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

def resize_and_crop(img, size, crop_type="center"):
    '''Resize and crop the image to the given size.'''
    if crop_type == "top":
        center = (0, 0)
    elif crop_type == "center":
        center = (0.5, 0.5)
    else:
        raise ValueError

    resize = list(size)
    if size[0] is None:
        resize[0] = img.size[0]
    if size[1] is None:
        resize[1] = img.size[1]
    return ImageOps.fit(img, resize, centering=center)

def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)[0]
    init_image = totensor(init_image)

    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

# --- Encoder Attack (Simple) ---
def pgd(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None, targets=None):
    """
    Projected Gradient Descent for Encoder Attack.
    If targets is None, maximizes the norm (untargeted attack/disruption).
    If targets is provided, minimizes distance to target (targeted attack).
    """
    device = X.device
    X_adv = X.clone().detach() + (torch.rand(*X.shape, device=device)*2*eps-eps)

    # We use a loop without tqdm if we want to suppress output, or pass a progress bar
    # For now, simple loop
    for i in range(iters):
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        X_adv.requires_grad_(True)

        if targets is not None:
            # Targeted attack: minimize distance to target
            loss = (model(X_adv).latent_dist.mean - targets).norm()
        else:
            # Untargeted: maximize norm (disrupt embedding)
            # Note: The notebook uses loss = norm(), and then does X_adv - grad.
            # This means it minimizes the norm?
            # Wait. "grad, = torch.autograd.grad(loss, [X_adv])"
            # "X_adv = X_adv - grad..." -> Gradient Descent.
            # So if loss is norm(), we are minimizing the norm?
            # Let's check demo_simple_attack_img2img.ipynb code:
            # loss = (model(X_adv).latent_dist.mean).norm()
            # X_adv = X_adv - grad...
            # Yes, minimizing the norm.
            # This makes the embedding close to zero?
            # Or maybe the logic is "minimize the norm of the embedding"?
            # Actually, let's look at demo/app.py.
            # It targets a gray image.
            # loss = (model... - target).norm()
            # Minimizing distance to gray image.

            # If targets is None, let's assume we want to maximize disruption.
            # Usually maximizing loss means X + grad.
            # But the code uses X - grad.
            # So we should define loss such that minimizing it achieves our goal.
            # If we want to move AWAY from original, we should Maximize distance.
            # But standard PGD minimizes a loss.
            # Let's stick to the notebook's implementation:
            # loss = norm(). X - grad. -> Minimize norm.
            # Maybe minimizing the norm makes it "empty"?
            loss = (model(X_adv).latent_dist.mean).norm()

        grad, = torch.autograd.grad(loss, [X_adv])

        # Gradient Descent (Minimize loss)
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        # Check constraints (PGD projection)
        # We want to clip X_adv such that it stays within X +/- eps
        # But wait, X has values in [-1, 1] (roughly, after preprocess).
        # And we want X_adv to be within eps of X in L_inf norm.

        # Clip perturbation to [-eps, eps]
        delta = X_adv - X
        delta = torch.clamp(delta, min=-eps, max=eps)
        X_adv = X + delta

        # Clip image to valid range [clamp_min, clamp_max]
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        if mask is not None:
            # Mask: 1 where we allow changes (perturbation), 0 where we preserve original.
            # X_adv = mask * X_adv + (1-mask) * X_orig
            # Assuming X is the original unperturbed image? No, X passed to pgd IS the starting point.
            # But we want to preserve the ORIGINAL pixels where mask=0.
            # So we need access to X_orig. PGD takes X.
            # In our usage X is X_orig (initial image).
            # So:
            X_adv.data = X_adv * mask + X * (1 - mask)

    return X_adv

# --- Diffusion Attack (Complex) ---
def attack_forward(
        self,
        prompt,
        masked_image,
        mask,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
    ):
    # self is the pipe_inpaint
    device = self.device

    # Encode text
    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=self.tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    text_embeddings = self.text_encoder(text_input_ids.to(device))[0]

    uncond_tokens = [""]
    max_length = text_input_ids.shape[-1]
    uncond_input = self.tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
    # seq_len = uncond_embeddings.shape[1] # unused
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    text_embeddings = text_embeddings.detach()

    num_channels_latents = self.vae.config.latent_channels

    latents_shape = (1 , num_channels_latents, height // 8, width // 8)
    latents = torch.randn(latents_shape, device=device, dtype=text_embeddings.dtype)

    # Resize mask to latents shape
    mask_resized = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
    mask_concatenated = torch.cat([mask_resized] * 2)

    masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
    masked_image_latents = 0.18215 * masked_image_latents
    masked_image_latents = torch.cat([masked_image_latents] * 2)

    latents = latents * self.scheduler.init_noise_sigma

    self.scheduler.set_timesteps(num_inference_steps)
    timesteps_tensor = self.scheduler.timesteps.to(device)

    for i, t in enumerate(timesteps_tensor):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = torch.cat([latent_model_input, mask_concatenated, masked_image_latents], dim=1)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample

    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample
    return image

def compute_grad(pipe, cur_mask, cur_masked_image, prompt, target_image, **kwargs):
    torch.set_grad_enabled(True)
    cur_mask = cur_mask.clone()
    cur_masked_image = cur_masked_image.clone()
    cur_mask.requires_grad = False
    cur_masked_image.requires_grad_()

    # We need to bind the attack_forward method or call it with self=pipe
    image_nat = attack_forward(pipe, mask=cur_mask,
                               masked_image=cur_masked_image,
                               prompt=prompt,
                               **kwargs)

    loss = (image_nat - target_image).norm(p=2)

    # If using CPU, ensure we don't error out on autograd if things aren't set up perfectly
    # But PyTorch autograd works on CPU.
    grad = torch.autograd.grad(loss, [cur_masked_image])[0] * (1 - cur_mask)

    return grad, loss.item(), image_nat.data.cpu()

def super_l2(pipe, cur_mask, X, prompt, step_size, iters, eps, clamp_min, clamp_max, grad_reps=5, target_image=0, **kwargs):
    X_adv = X.clone()

    # Using a simple range to avoid tqdm spam in logs, or use tqdm if interactive
    iterator = range(iters)

    for i in iterator:
        all_grads = []
        losses = []
        for j in range(grad_reps):
            c_grad, loss, last_image = compute_grad(pipe, cur_mask, X_adv, prompt, target_image=target_image, **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)

        if not all_grads:
             break

        grad = torch.stack(all_grads).mean(0)

        # Log loss occasionally?
        # print(f'Step {i}, AVG Loss: {np.mean(losses):.3f}')

        l = len(X.shape) - 1
        grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
        grad_normalized = grad.detach() / (grad_norm + 1e-10)

        actual_step_size = step_size
        X_adv = X_adv - grad_normalized * actual_step_size

        d_x = X_adv - X.detach()
        d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
        X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return X_adv
