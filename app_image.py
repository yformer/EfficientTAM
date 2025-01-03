from typing import Dict, Optional, Tuple

import gradio as gr
import numpy as np
import supervision as sv
import torch

from efficient_track_anything.automatic_mask_generator import (
    EfficientTAMAutomaticMaskGenerator,
)
from efficient_track_anything.build_efficienttam import build_efficienttam
from efficient_track_anything.efficienttam_image_predictor import (
    EfficientTAMImagePredictor,
)
from gradio_image_prompter import ImagePrompter
from PIL import Image


MARKDOWN = """
# Efficient Track Anything Image🔥
<div>
    <a href="https://github.com/yformer/EfficientTAM">
        <img src="https://badges.aleen42.com/src/github.svg" alt="GitHub" style="display:inline-block;">
    </a>
    <a href="https://www.youtube.com/watch?v=SsuDcx4ZiVQ">
        <img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube" style="display:inline-block;">
    </a>
</div>
Efficient Track Anything is an efficient foundation model for promptable image and video segmentation. 

Video segmentation is available at [efficient track anything](https://yformer.github.io/efficient-track-anything/).

Our track anything image demo is built on [Piotr Skalski
's demo](https://huggingface.co/spaces/SkalskiP/segment-anything-model-2)
"""
BOX_PROMPT_MODE = "box prompt"
MASK_GENERATION_MODE = "mask generation"
MODE_NAMES = [BOX_PROMPT_MODE, MASK_GENERATION_MODE]

EXAMPLES = [
    ["efficienttam-s", MASK_GENERATION_MODE, "examples/mario_1.jpg", None],
    ["efficienttam-s", MASK_GENERATION_MODE, "examples/sf.jpg", None],
    ["efficienttam-s", MASK_GENERATION_MODE, "examples/toy.jpg", None],
    ["efficienttam-s", MASK_GENERATION_MODE, "examples/mario_2.jpg", None],
    ["efficienttam-s", MASK_GENERATION_MODE, "examples/bill.jpg", None],
]

DEVICE = "cuda"

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

CHECKPOINT_NAMES = ["efficienttam-s", "efficienttam-ti"]
CHECKPOINTS = {
    "efficienttam-s": [
        "configs/efficienttam/efficienttam_s.yaml",
        "./checkpoints/efficienttam_s.pt",
    ],
    "efficienttam-ti": [
        "configs/efficienttam/efficienttam_ti.yaml",
        "./checkpoints/efficienttam_ti.pt",
    ],
}


def load_models(
    device: torch.device,
) -> Tuple[
    Dict[str, EfficientTAMImagePredictor], Dict[str, EfficientTAMAutomaticMaskGenerator]
]:
    image_predictors = {}
    mask_generators = {}
    for key, (config, checkpoint) in CHECKPOINTS.items():
        model = build_efficienttam(config, checkpoint)
        image_predictors[key] = EfficientTAMImagePredictor(efficienttam_model=model)
        mask_generators[key] = EfficientTAMAutomaticMaskGenerator(
            model=model,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
        )
    return image_predictors, mask_generators


MASK_ANNOTATOR = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
IMAGE_PREDICTORS, MASK_GENERATORS = load_models(device=DEVICE)


def process(
    checkpoint_dropdown,
    mode_dropdown,
    image_input,
    image_prompter_input,
) -> Optional[Image.Image]:
    if (image_input is None) and (image_prompter_input is None):
        return image_input

    if mode_dropdown == BOX_PROMPT_MODE:
        if image_prompter_input is None:
            return None
        image_input = image_prompter_input["image"]
        if image_input is None:
            return image_input
        prompt = image_prompter_input["points"]
        if len(prompt) == 0:
            return image_input

        model = IMAGE_PREDICTORS[checkpoint_dropdown]
        image = np.array(image_input.convert("RGB"))
        box = np.array([[x1, y1, x2, y2] for x1, y1, _, x2, y2, _ in prompt])

        model.set_image(image)
        masks, _, _ = model.predict(box=box, multimask_output=False)

        # dirty fix; remove this later
        if len(masks.shape) == 4:
            masks = np.squeeze(masks)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks), mask=masks.astype(bool)
        )
        return MASK_ANNOTATOR.annotate(image_input, detections)

    if mode_dropdown == MASK_GENERATION_MODE:
        model = MASK_GENERATORS[checkpoint_dropdown]
        if image_input is None:
            return image_input
        image_input.visibility = True
        image = np.array(image_input.convert("RGB"))
        result = model.generate(image)
        detections = sv.Detections.from_sam(result)
        return MASK_ANNOTATOR.annotate(image_input, detections)


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        checkpoint_dropdown_component = gr.Dropdown(
            choices=CHECKPOINT_NAMES,
            value=CHECKPOINT_NAMES[0],
            label="Checkpoint",
            info="Select a efficient track anything checkpoint to use.",
            interactive=True,
        )
        mode_dropdown_component = gr.Dropdown(
            choices=MODE_NAMES,
            value=MODE_NAMES[1],
            label="Mode",
            info="Select a mode to use. `box prompt` if you want to generate masks for "
            "selected objects, `mask generation` if you want to generate masks "
            "for the whole image.",
            interactive=True,
        )
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(type="pil", label="Upload image")
            image_prompter_input_component = ImagePrompter(
                type="pil", label="Image prompt", visible=False
            )
            submit_button_component = gr.Button(value="Submit", variant="primary")
        with gr.Column():
            image_output_component = gr.Image(type="pil", label="Image Output")
    with gr.Row():
        gr.Examples(
            examples=EXAMPLES,
            inputs=[
                checkpoint_dropdown_component,
                mode_dropdown_component,
                image_input_component,
                image_prompter_input_component,
            ],
        )

    def on_mode_dropdown_change(text):
        return [
            ImagePrompter(visible=(text == BOX_PROMPT_MODE)),
            gr.Image(visible=(text == MASK_GENERATION_MODE)),
        ]

    mode_dropdown_component.change(
        on_mode_dropdown_change,
        inputs=[mode_dropdown_component],
        outputs=[image_prompter_input_component, image_input_component],
        queue=False,
    )
    submit_button_component.click(
        fn=process,
        inputs=[
            checkpoint_dropdown_component,
            mode_dropdown_component,
            image_input_component,
            image_prompter_input_component,
        ],
        outputs=[image_output_component],
        queue=False,
    )

demo.queue()
demo.launch(share=True)
