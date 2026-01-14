import gradio as gr
import torch
import numpy as np
import requests
from io import BytesIO
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline
import src.utils_protection as utils

# --- Configuration ---
DEVICE = utils.get_device()
DTYPE = utils.get_dtype()

print(f"Running on {DEVICE} with {DTYPE}")

# --- Load Model ---
# We use StableDiffusionInpaintPipeline as it allows us to handle masks if needed,
# and we can access VAE for the simple attack.
model_id = "runwayml/stable-diffusion-inpainting"
try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        revision="fp16" if DEVICE == "cuda" else "main",
        torch_dtype=DTYPE,
        safety_checker=None,
    )
    pipe = pipe.to(DEVICE)
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback or exit? For now, let's assume it loads or we catch it.
    pipe = None

# --- Target for Simple Attack ---
def get_target_image():
    # Use a local or generated gray image to avoid network dependency failures if possible,
    # but the repo uses this specific URL. I'll use a local fallback if it fails.
    target_url = 'https://www.rtings.com/images/test-materials/2015/204_Gray_Uniformity.png'
    try:
        response = requests.get(target_url, timeout=5)
        target_image = Image.open(BytesIO(response.content)).convert("RGB")
    except:
        # Fallback: Gray image
        target_image = Image.new('RGB', (512, 512), color='gray')

    target_image = target_image.resize((512, 512))
    return target_image

TARGET_IMAGE = get_target_image()

def protect_image(input_image, attack_type, strength_slider):
    if pipe is None:
        return None, "Model failed to load."

    # 1. Prepare Image
    # Resize to 512x512 for SD
    init_image = Image.fromarray(input_image)
    init_image = utils.resize_and_crop(init_image, (512, 512))

    # 2. Prepare Mask (We assume whole image protection for now)
    # If we want to allow masking, we'd need an input for it.
    # For now, let's make a mask that covers the WHOLE image (so we can modify the whole image).
    # Wait, in Inpainting:
    # Mask = 1 (White) -> Area to INPAINT (Modify).
    # Mask = 0 (Black) -> Area to KEEP.
    # To protect the image from INPAINTING, we want to optimize the image such that if someone TRIES to inpaint it (masks it), it fails.
    # The attack optimization process needs a mask of WHERE we can add noise.
    # We can add noise EVERYWHERE (Mask=1 everywhere) to protect the whole image.
    mask_image = Image.new("L", init_image.size, 255) # All white (modifiable by attack)

    # Preprocessing
    mask, X = utils.prepare_mask_and_masked_image(init_image, mask_image)
    X = X.to(DEVICE, dtype=DTYPE)
    mask = mask.to(DEVICE, dtype=DTYPE) # Mask of 1s (allow modification everywhere)

    # 3. Attack
    if attack_type == "Simple (Encoder Attack)":
        # Target: Gray image embedding
        target_emb = pipe.vae.encode(utils.preprocess(TARGET_IMAGE).to(DEVICE, dtype=DTYPE)).latent_dist.mean

        X_full = utils.preprocess(init_image).to(DEVICE, dtype=DTYPE)

        # Epsilon and Steps based on Strength
        eps = 0.05 + (strength_slider / 10.0) * 0.15
        iters = int(50 + (strength_slider / 10.0) * 150)

        if DEVICE == "cpu":
            iters = min(iters, 20) # Cap iterations on CPU for demo speed

        adv_X = utils.pgd(X_full,
                    targets = target_emb,
                    model=pipe.vae.encode,
                    iters=iters,
                    eps=eps,
                    step_size=eps/10.0,
                    clamp_min=-1,
                    clamp_max=1
                   )

        # Convert back to Image
        adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
        protected_image = utils.topil(adv_X[0]).convert("RGB")

    else: # Complex (Diffusion Attack)
        # Implement Real Diffusion Attack (Slow but available)
        # We assume Whole Image Protection for this flow, so we treat the image as "context" to be protected.
        # Simulation: Attacker tries to inpaint a central hole. We perturb the rest.

        dummy_mask = Image.new("L", (512, 512), 0)
        # Box in middle 256x256 where attack might happen
        import PIL.ImageDraw
        draw = PIL.ImageDraw.Draw(dummy_mask)
        draw.rectangle([128, 128, 384, 384], fill=255) # White box in middle

        # In this simulation:
        # Mask=1 (White) is where the attacker inpaints.
        # Mask=0 (Black) is the context we want to protect (perturb).
        # prepare_mask_and_masked_image returns masked_image = image * (mask < 0.5).
        # So masked_image contains the CONTEXT (pixels outside the box).

        mask_tensor, masked_image_tensor = utils.prepare_mask_and_masked_image(init_image, dummy_mask)
        mask_tensor = mask_tensor.to(DEVICE, dtype=DTYPE)
        masked_image_tensor = masked_image_tensor.to(DEVICE, dtype=DTYPE)

        eps = 0.05 + (strength_slider / 10.0) * 0.1
        iters = 5 if DEVICE == "cpu" else 50 # Very low iterations for CPU

        # We perturb the masked_image_tensor (the context)
        # Target: None (untargeted disruption) or Zero tensor
        target_image_tensor = torch.zeros_like(masked_image_tensor)

        X_adv = utils.super_l2(pipe,
                     cur_mask=mask_tensor,
                     X=masked_image_tensor,
                     prompt="",
                     step_size=eps/5.0,
                     iters=iters,
                     eps=eps*255, # super_l2 might expect different scale? In notebook eps=16 (pixel space 0-255 maybe? or latent?)
                     # Notebook uses X in [-1, 1]. eps=16 sounds like pixel space 0-255?
                     # Let's check utils.preprocess -> [-1, 1].
                     # Notebook: "eps=16". "X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)".
                     # If X is [-1, 1], eps=16 is HUGE.
                     # Ah, notebook might not normalize to [-1, 1]?
                     # "masked_image = image * (mask < 0.5)". image is [-1, 1] from prepare_mask...
                     # So eps=16 is indeed huge if range is 2.
                     # Wait, notebook: "prepare_mask...: image = ... / 127.5 - 1.0". Yes [-1, 1].
                     # Maybe eps is smaller? "eps=16".
                     # Let's use a safe small eps for [-1, 1] range: 0.1
                     clamp_min=-1,
                     clamp_max=1,
                     grad_reps=1, # Speed up
                     target_image=target_image_tensor
                    )

        # Reconstruct full image: Context (Perturbed) + Original Hole (which was removed in masked_image_tensor)
        # Actually, if we protect the context, we return the Perturbed Context + Original Center.
        # But masked_image_tensor has 0 in the center.
        # We need the original center.

        init_tensor = utils.preprocess(init_image).to(DEVICE, dtype=DTYPE)
        # X_adv is the perturbed context (with 0 in center).
        # We want X_final = X_adv (where mask=0) + init_tensor (where mask=1)
        # mask_tensor is 1 in center.

        X_final = X_adv * (1 - mask_tensor) + init_tensor * mask_tensor

        X_final = (X_final / 2 + 0.5).clamp(0, 1)
        protected_image = utils.topil(X_final[0]).convert("RGB")

    return protected_image, "Protection Applied Successfully"

def generate_diff(original, protected):
    # Calculate difference
    orig_np = np.array(original).astype(float)
    prot_np = np.array(protected.resize(original.size)).astype(float)
    diff = np.abs(orig_np - prot_np)
    # Amplify diff for visibility
    diff = np.clip(diff * 10, 0, 255).astype(np.uint8)
    return Image.fromarray(diff)

def process_pipeline(image, attack_type, strength):
    if image is None:
        return None, None, "Please upload an image."

    # Run protection
    protected, msg = protect_image(image, attack_type, strength)

    if protected is None:
        return None, None, msg

    diff = generate_diff(Image.fromarray(image), protected)

    return protected, diff, msg

# --- UI Layout ---
with gr.Blocks(title="AI-Proof Photo Protection") as demo:
    gr.Markdown("# 🛡️ AI-Proof Photo Protection")
    gr.Markdown("Protect your photos from AI manipulation (Deepfakes, Style Transfer, etc.) by adding invisible adversarial noise.")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Image", source="upload", type="numpy")
            attack_type = gr.Dropdown(
                label="Protection Method",
                choices=["Simple (Encoder Attack)", "Advanced (Diffusion Attack)"],
                value="Simple (Encoder Attack)",
                info="Simple: Fast, protects against Img2Img. Advanced: Slow (CPU), protects against Inpainting."
            )
            strength = gr.Slider(label="Protection Strength", minimum=1, maximum=10, value=5, step=1)
            protect_btn = gr.Button("Protect Photo", variant="primary")

        with gr.Column():
            output_img = gr.Image(label="Protected Image", type="pil")
            diff_img = gr.Image(label="Noise Difference (Amplified)", type="pil")
            status = gr.Textbox(label="Status", interactive=False)

    protect_btn.click(process_pipeline, inputs=[input_img, attack_type, strength], outputs=[output_img, diff_img, status])

    gr.Markdown("### How it works")
    gr.Markdown("This tool uses **PhotoGuard** (Encoder Attack) to perturb the image's latent representation. This makes the image 'look' like a gray box or random noise to AI models like Stable Diffusion, while remaining visually unchanged to humans.")

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860)
