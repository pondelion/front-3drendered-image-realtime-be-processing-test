from typing import List, Optional, Union

import torch
from diffusers import AutoPipelineForInpainting, LCMScheduler
from transformers import pipeline
from PIL.Image import Image as PILImage
import numpy as np


owlvit = None
stream_diffusion_img2img = None
mobile_sam = None
mobile_sam_mask_generator = None
mobile_sam_predictor = None
sd_inpaint_lcmlora = None


def owlvit_detect(image: PILImage, labels: List[str]):
    global owlvit
    if owlvit is None:
        owlvit = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection", device=0)
    preds = owlvit(image, candidate_labels=labels)
    # [
    #     {'score': 0.3571370542049408,
    #      'label': 'human',
    #       'box': {'xmin': 180, 'ymin': 71, 'xmax': 271, 'ymax': 178}},
    #     ...
    # ]
    return preds


def apply_stream_diffusion_img2img(
    image: PILImage, prompt: str
) -> Union[PILImage, List[PILImage]]:
    global stream_diffusion_img2img
    guidance_scale: float = 1.2
    negative_prompt: str = "low quality, bad quality, blurry, low resolution"
    delta: float = 0.5
    if stream_diffusion_img2img is None:
        print('initializing stream diffusion img2img..')
        from StreamDiffusion.utils.wrapper import StreamDiffusionWrapper
        model_id_or_path: str = "KBlueLeaf/kohaku-v2.1"
        lora_dict = None
        width: int = 512
        height: int = 512
        acceleration = "xformers"
        use_denoising_batch: bool = True
        cfg_type = "self"
        seed: int = 2
        if guidance_scale <= 1.0:
            cfg_type = "none"
        stream_diffusion_img2img = StreamDiffusionWrapper(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            t_index_list=[22, 32, 45],
            frame_buffer_size=1,
            width=width,
            height=height,
            warmup=10,
            acceleration=acceleration,
            mode="img2img",
            use_denoising_batch=use_denoising_batch,
            cfg_type=cfg_type,
            seed=seed,
        )
    # stream_diffusion_img2img.prepare(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     num_inference_steps=50,
    #     guidance_scale=guidance_scale,
    #     delta=delta,
    # )
    # image_tensor = stream_diffusion_img2img.preprocess_image(image)
    # for _ in range(stream_diffusion_img2img.batch_size - 1):
    #     stream_diffusion_img2img(image=image_tensor)
    # output_image = stream_diffusion_img2img(image=image_tensor)
    output_image = stream_diffusion_img2img.img2img(image, prompt=prompt)
    return output_image


def mobilesam_detect(
    image: PILImage,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    device="cuda"
):
    global mobile_sam
    global mobile_sam_mask_generator
    global mobile_sam_predictor
    if mobile_sam is None:
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        model_type = "vit_t"
        sam_checkpoint = "MobileSAM/weights/mobile_sam.pt"
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.eval()
        mobile_sam.to(device=device)
        mobile_sam_mask_generator = SamAutomaticMaskGenerator(mobile_sam)
        mobile_sam_predictor = SamPredictor(mobile_sam)

    if points is None:
        assert mobile_sam_mask_generator is not None
        masks = mobile_sam_mask_generator.generate(image)
        return masks
    else:
        assert mobile_sam_predictor is not None
        masks, scores, logits = mobile_sam_predictor.predict(
            point_coords=points, point_labels=labels, multimask_output=True,
        )
        return masks, scores, logits


def apply_sd_inpaint(
    image: PILImage,
    mask: PILImage,
    prompt: str,
    num_inference_steps: int = 3,
    guidance_scale: int = 4,
) -> PILImage:
    global sd_inpaint_lcmlora
    if sd_inpaint_lcmlora is None:
        sd_inpaint_lcmlora = AutoPipelineForInpainting.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        # set scheduler
        sd_inpaint_lcmlora.scheduler = LCMScheduler.from_config(
            sd_inpaint_lcmlora.scheduler.config
        )
        # load LCM-LoRA
        sd_inpaint_lcmlora.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        sd_inpaint_lcmlora.set_progress_bar_config(leave=False)
    generator = torch.manual_seed(0)
    image_gen = sd_inpaint_lcmlora(
        prompt=prompt,
        image=image,
        mask_image=mask,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    return image_gen
