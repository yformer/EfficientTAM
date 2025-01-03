import os
import re
import subprocess
from datetime import datetime
from typing import List, Optional, Tuple

import gradio as gr

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0,1,2,3,4,5,6,7"
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)

from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageFilter

# Description
title = "<center><strong><font size='8'>Efficient Track Anything (EfficientTAM)<font></strong></center>"

description_e = """This is a demo of [Efficient Track Anything (EfficientTAM) Model](https://github.com/yformer/EfficientTAM).
              """

description_p = """# Efficient Track Anything
                - Built our demo based on [SAM2-Video-Predictor](https://huggingface.co/spaces/fffiloni/SAM2-Video-Predictor). Thanks to Sylvain Filoni.
                - Instruction
                <ol>
                <li> Download [`🤗Efficient Track Anything Checkpoints`](https://huggingface.co/yunyangx/efficient-track-anything/tree/main) </li>
                <li> Upload one video or click one example video</li>
                <li> Click 'include' point type, select the object to segment and track</li>
                <li> Click 'exclude' point type (optional), select the area you want to avoid segmenting and tracking</li>
                <li> Click the 'Segment' button, obtain the mask of the first frame </li>
                <li> Click the 'coarse' level and the 'Track' button, segment and track the object every 15 frames </li>
                <li> Click the corresponding frame to add points on the object for mask refining (optional) </li>
                <li> Click the 'fine' level and the 'Track' button, obtain masklet and masked video </li>
                <li> Click the 'Reset' button to restart </li>
                </ol>
                - Github [link](https://github.com/yformer/EfficientTAM)
                - [`🤗Efficient Track Anything Checkpoints`](https://huggingface.co/yunyangx/efficient-track-anything/tree/main)
              """

# examples
examples = [
    ["examples/videos/cat.mp4"],
    ["examples/videos/coffee.mp4"],
    ["examples/videos/car.mp4"],
    ["examples/videos/chick.mp4"],
    ["examples/videos/cups.mp4"],
    ["examples/videos/dog.mp4"],
    ["examples/videos/goat.mp4"],
    ["examples/videos/juggle.mp4"],
    ["examples/videos/street.mp4"],
    ["examples/videos/yacht.mp4"],
]

default_example = examples[0]


def get_video_fps(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    return fps


def clear_points(image):
    # we clean all
    return [
        image,  # first_frame_path
        gr.State([]),  # tracking_points
        gr.State([]),  # trackings_input_label
        image,  # points_map
        # gr.State()     # stored_inference_state
    ]


def preprocess_video_in(video_path):
    if video_path is None:
        return (
            None,
            gr.State([]),
            gr.State([]),
            None,
            None,
            None,
            None,
            None,
            None,
            gr.update(open=True),
        )

    # Generate a unique ID based on the current date and time
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

    # Set directory with this ID to store video frames
    extracted_frames_output_dir = f"frames_{unique_id}"

    # Create the output directory
    os.makedirs(extracted_frames_output_dir, exist_ok=True)

    ### Process video frames ###
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the number of frames to process (10 seconds of video)
    max_frames = int(fps * 10)

    frame_number = 0
    first_frame = None

    while True:
        ret, frame = cap.read()
        if not ret or frame_number >= max_frames:
            break

        # Format the frame filename as '00000.jpg'
        frame_filename = os.path.join(
            extracted_frames_output_dir, f"{frame_number:05d}.jpg"
        )

        # Save the frame as a JPEG file
        cv2.imwrite(frame_filename, frame)

        # Store the first frame
        if frame_number == 0:
            first_frame = frame_filename

        frame_number += 1

    # Release the video capture object
    cap.release()

    # scan all the JPEG frame names in this directory
    scanned_frames = [
        p
        for p in os.listdir(extracted_frames_output_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    scanned_frames.sort(key=lambda p: int(os.path.splitext(p)[0]))
    # print(f"SCANNED_FRAMES: {scanned_frames}")

    return [
        first_frame,  # first_frame_path
        gr.State([]),  # tracking_points
        gr.State([]),  # trackings_input_label
        first_frame,  # input_first_frame_image
        first_frame,  # points_map
        extracted_frames_output_dir,  # video_frames_dir
        scanned_frames,  # scanned_frames
        None,  # stored_inference_state
        None,  # stored_frame_names
        gr.update(open=False),  # video_in_drawer
    ]


def get_point(
    point_type,
    tracking_points,
    trackings_input_label,
    input_first_frame_image,
    evt: gr.SelectData,
):
    if input_first_frame_image is None:
        return gr.State([]), gr.State([]), None
    print(f"You selected {evt.value} at {evt.index} from {evt.target}")

    tracking_points.value.append(evt.index)
    print(f"TRACKING POINT: {tracking_points.value}")

    if point_type == "include":
        trackings_input_label.value.append(1)
    elif point_type == "exclude":
        trackings_input_label.value.append(0)
    print(f"TRACKING INPUT LABEL: {trackings_input_label.value}")

    # Open the image and get its dimensions
    transparent_background = Image.open(input_first_frame_image).convert("RGBA")
    w, h = transparent_background.size

    # Define the circle radius as a fraction of the smaller dimension
    fraction = 0.02  # You can adjust this value as needed
    radius = int(fraction * min(w, h))

    # Create a transparent layer to draw on
    transparent_layer = np.zeros((h, w, 4), dtype=np.uint8)

    for index, track in enumerate(tracking_points.value):
        if trackings_input_label.value[index] == 1:
            cv2.circle(transparent_layer, track, radius, (0, 255, 0, 255), -1)
        else:
            cv2.circle(transparent_layer, track, radius, (255, 0, 0, 255), -1)

    # Convert the transparent layer back to an image
    transparent_layer = Image.fromarray(transparent_layer, "RGBA")
    selected_point_map = Image.alpha_composite(
        transparent_background, transparent_layer
    )

    return tracking_points, trackings_input_label, selected_point_map


DEVICE = "cuda"
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.axis("off")
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def load_model(checkpoint):
    # Load model accordingly to user's choice
    if checkpoint == "efficienttam_s":
        efficienttam_checkpoint = "./checkpoints/efficienttam_s.pt"
        model_cfg = "configs/efficienttam/efficienttam_s.yaml"
        return [efficienttam_checkpoint, model_cfg]
    elif checkpoint == "efficienttam_ti":
        efficienttam_checkpoint = "./checkpoints/efficienttam_ti.pt"
        model_cfg = "configs/efficienttam/efficienttam_ti.yaml"
        return [efficienttam_checkpoint, model_cfg]
    elif checkpoint == "efficienttam_s_512x512":
        efficienttam_checkpoint = "./checkpoints/efficienttam_s_512x512.pt"
        model_cfg = "configs/efficienttam/efficienttam_s_512x512.yaml"
        return [efficienttam_checkpoint, model_cfg]
    elif checkpoint == "efficienttam_ti_512x512":
        efficienttam_checkpoint = "./checkpoints/efficienttam_ti_512x512.pt"
        model_cfg = "configs/efficienttam/efficienttam_ti_512x512.yaml"
        return [efficienttam_checkpoint, model_cfg]
    elif checkpoint == "efficienttam_s_1":
        efficienttam_checkpoint = "./checkpoints/efficienttam_s_1.pt"
        model_cfg = "configs/efficienttam/efficienttam_s_1.yaml"
        return [efficienttam_checkpoint, model_cfg]
    elif checkpoint == "efficienttam_s_2":
        efficienttam_checkpoint = "./checkpoints/efficienttam_s_2.pt"
        model_cfg = "configs/efficienttam/efficienttam_s_2.yaml"
        return [efficienttam_checkpoint, model_cfg]
    elif checkpoint == "efficienttam_ti_1":
        efficienttam_checkpoint = "./checkpoints/efficienttam_ti_1.pt"
        model_cfg = "configs/efficienttam/efficienttam_ti_1.yaml"
        return [efficienttam_checkpoint, model_cfg]
    elif checkpoint == "efficienttam_ti_2":
        efficienttam_checkpoint = "./checkpoints/efficienttam_ti_2.pt"
        model_cfg = "configs/efficienttam/efficienttam_ti_2.yaml"
        return [efficienttam_checkpoint, model_cfg]
    else:
        efficienttam_checkpoint = "./checkpoints/efficienttam_s.pt"
        model_cfg = "configs/efficienttam/efficienttam_s.yaml"
        return [efficienttam_checkpoint, model_cfg]


def get_mask_efficienttam_process(
    stored_inference_state,
    input_first_frame_image,
    checkpoint,
    tracking_points,
    trackings_input_label,
    video_frames_dir,  # extracted_frames_output_dir defined in 'preprocess_video_in' function
    scanned_frames,
    working_frame: str = None,  # current frame being added points
    available_frames_to_check: List[str] = [],
):

    if len(tracking_points.value) == 0:
        return (
            gr.update(visible=False),
            None,
            gr.State(),
            None,
            stored_inference_state,
            working_frame,
        )
    # get model and model config paths
    print(f"USER CHOSEN CHECKPOINT: {checkpoint}")
    efficienttam_checkpoint, model_cfg = load_model(checkpoint)
    print("MODEL LOADED")

    # set predictor
    predictor = build_efficienttam_video_predictor(
        model_cfg, efficienttam_checkpoint, device="cuda"
    )
    print("PREDICTOR READY")

    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
    # print(f"STATE FRAME OUTPUT DIRECTORY: {video_frames_dir}")
    video_dir = video_frames_dir

    # scan all the JPEG frame names in this directory
    frame_names = scanned_frames

    # print(f"STORED INFERENCE STEP: {stored_inference_state}")
    if stored_inference_state is None:
        # Init inference_state
        inference_state = predictor.init_state(video_path=video_dir)
        print("NEW INFERENCE_STATE INITIATED")
    else:
        inference_state = stored_inference_state

    # segment and track one object
    # predictor.reset_state(inference_state) # if any previous tracking, reset

    ### HANDLING WORKING FRAME
    # new_working_frame = None
    # Add new point
    if working_frame is None:
        ann_frame_idx = (
            0  # the frame index we interact with, 0 if it is the first frame
        )
        working_frame = "frame_0.jpg"
    else:
        # Use a regular expression to find the integer
        match = re.search(r"frame_(\d+)", working_frame)
        if match:
            # Extract the integer from the match
            frame_number = int(match.group(1))
            ann_frame_idx = frame_number

    print(f"NEW_WORKING_FRAME PATH: {working_frame}")

    ann_obj_id = (
        1  # give a unique id to each object we interact with (it can be any integers)
    )

    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array(tracking_points.value, dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array(trackings_input_label.value, np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask(
        (out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0]
    )

    # Save the plot as a JPG file
    first_frame_output_filename = "output_first_frame.jpg"
    plt.savefig(first_frame_output_filename, format="jpg")
    plt.close()
    torch.cuda.empty_cache()

    # Assuming available_frames_to_check.value is a list
    if working_frame not in available_frames_to_check:
        available_frames_to_check.append(working_frame)
        print(available_frames_to_check)

    return (
        gr.update(visible=True),
        "output_first_frame.jpg",
        frame_names,
        predictor,
        inference_state,
        gr.update(choices=available_frames_to_check, value=working_frame, visible=True),
    )


def propagate_to_all(
    tracking_points,
    video_in,
    checkpoint,
    stored_inference_state,
    stored_frame_names,
    video_frames_dir,
    vis_frame_type,
    available_frames_to_check,
    working_frame,
):
    if (
        tracking_points is None
        or video_in is None
        or checkpoint is None
        or stored_inference_state is None
    ):
        return (
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            available_frames_to_check,
            gr.update(visible=False),
        )
    #### PROPAGATION ####
    efficienttam_checkpoint, model_cfg = load_model(checkpoint)
    predictor = build_efficienttam_video_predictor(
        model_cfg, efficienttam_checkpoint, device="cuda"
    )

    inference_state = stored_inference_state
    frame_names = stored_frame_names
    video_dir = video_frames_dir

    # Define a directory to save the JPEG images
    frames_output_dir = "frames_output_images"
    os.makedirs(frames_output_dir, exist_ok=True)

    # Initialize a list to store file paths of saved images
    jpeg_images = []

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    # for out_frame_idx, out_obj_ids, out_mask_logits in g_predictor.propagate_in_video(inference_state):
    #     video_segments[out_frame_idx] = {
    #         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
    #         for i, out_obj_id in enumerate(out_obj_ids)
    #     }
    print("starting propagate_in_video")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    if vis_frame_type == "coarse":
        vis_frame_stride = 15
    elif vis_frame_type == "fine":
        vis_frame_stride = 1

    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        # Define the output filename and save the figure as a JPEG file
        output_filename = os.path.join(frames_output_dir, f"frame_{out_frame_idx}.jpg")
        plt.savefig(output_filename, format="jpg")

        # Close the plot
        plt.close()

        # Append the file path to the list
        jpeg_images.append(output_filename)

        if f"frame_{out_frame_idx}.jpg" not in available_frames_to_check:
            available_frames_to_check.append(f"frame_{out_frame_idx}.jpg")

    torch.cuda.empty_cache()
    print(f"JPEG_IMAGES: {jpeg_images}")

    if vis_frame_type == "coarse":
        return (
            gr.update(value=jpeg_images),
            gr.update(value=None),
            gr.update(
                choices=available_frames_to_check, value=working_frame, visible=True
            ),
            available_frames_to_check,
            gr.update(visible=True),
        )
    elif vis_frame_type == "fine":
        # Create a video clip from the image sequence
        original_fps = get_video_fps(video_in)
        fps = original_fps  # Frames per second
        total_frames = len(jpeg_images)
        clip = ImageSequenceClip(jpeg_images, fps=fps)
        # Write the result to a file
        final_vid_output_path = "output_video.mp4"

        # Write the result to a file
        clip.write_videofile(final_vid_output_path, codec="libx264")

        return (
            gr.update(value=None),
            gr.update(value=final_vid_output_path),
            working_frame,
            available_frames_to_check,
            gr.update(visible=True),
        )


def update_ui(vis_frame_type):
    if vis_frame_type == "coarse":
        return gr.update(visible=True), gr.update(visible=False)
    elif vis_frame_type == "fine":
        return gr.update(visible=False), gr.update(visible=True)


def switch_working_frame(working_frame, scanned_frames, video_frames_dir):
    new_working_frame = None
    if working_frame == None:
        new_working_frame = os.path.join(video_frames_dir, scanned_frames[0])

    else:
        # Use a regular expression to find the integer
        match = re.search(r"frame_(\d+)", working_frame)
        if match:
            # Extract the integer from the match
            frame_number = int(match.group(1))
            ann_frame_idx = frame_number
            new_working_frame = os.path.join(
                video_frames_dir, scanned_frames[ann_frame_idx]
            )
    return gr.State([]), gr.State([]), new_working_frame, new_working_frame


def reset_propagation(first_frame_path, predictor, stored_inference_state):

    predictor.reset_state(stored_inference_state)
    # print(f"RESET State: {stored_inference_state} ")
    return (
        first_frame_path,
        gr.State([]),
        gr.State([]),
        gr.update(value=None, visible=False),
        stored_inference_state,
        None,
        ["frame_0.jpg"],
        first_frame_path,
        "frame_0.jpg",
        gr.update(visible=False),
    )


with gr.Blocks() as demo:
    first_frame_path = gr.State()
    tracking_points = gr.State([])
    trackings_input_label = gr.State([])
    video_frames_dir = gr.State()
    scanned_frames = gr.State()
    loaded_predictor = gr.State()
    stored_inference_state = gr.State()
    stored_frame_names = gr.State()
    available_frames_to_check = gr.State([])
    with gr.Column():
        # Title
        gr.Markdown(title)
        with gr.Row():

            with gr.Column():
                # Instructions
                gr.Markdown(description_p)

                with gr.Accordion("Input Video", open=True) as video_in_drawer:
                    video_in = gr.Video(label="Input Video", format="mp4")

                with gr.Row():
                    point_type = gr.Radio(
                        label="point type",
                        choices=["include", "exclude"],
                        value="include",
                        scale=2,
                    )
                    clear_points_btn = gr.Button("Clear Points", scale=1)

                input_first_frame_image = gr.Image(
                    label="input image",
                    interactive=False,
                    type="filepath",
                    visible=False,
                )

                points_map = gr.Image(
                    label="Frame with Point Prompt", type="filepath", interactive=False
                )

                with gr.Row():
                    checkpoint = gr.Dropdown(
                        label="Checkpoint",
                        choices=[
                            "efficienttam_s",
                            "efficienttam_ti",
                            "efficienttam_s_512x512",
                            "efficienttam_ti_512x512",
                            "efficienttam_s_1",
                            "efficienttam_s_2",
                            "efficienttam_ti_1",
                            "efficienttam_ti_2",
                        ],
                        value="efficienttam_s",
                    )
                    submit_btn = gr.Button("Segment", size="lg")

            with gr.Column():
                gr.Markdown("# Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[
                        video_in,
                    ],
                )
                gr.Markdown("\n\n\n\n\n\n\n\n\n\n\n")
                gr.Markdown("\n\n\n\n\n\n\n\n\n\n\n")
                gr.Markdown("\n\n\n\n\n\n\n\n\n\n\n")
                with gr.Row():
                    working_frame = gr.Dropdown(
                        label="Frame ID",
                        choices=[""],
                        value=None,
                        visible=False,
                        allow_custom_value=False,
                        interactive=True,
                    )
                    change_current = gr.Button("change current", visible=False)
                output_result = gr.Image(label="Reference Mask")
                with gr.Row():
                    vis_frame_type = gr.Radio(
                        label="Track level",
                        choices=["coarse", "fine"],
                        value="coarse",
                        scale=2,
                    )
                    propagate_btn = gr.Button("Track", scale=1)
                reset_prpgt_brn = gr.Button("Reset", visible=False)
                output_propagated = gr.Gallery(
                    label="Masklets", columns=4, visible=False
                )
                output_video = gr.Video(visible=False)

    # When new video is uploaded
    video_in.upload(
        fn=preprocess_video_in,
        inputs=[video_in],
        outputs=[
            first_frame_path,
            tracking_points,  # update Tracking Points in the gr.State([]) object
            trackings_input_label,  # update Tracking Labels in the gr.State([]) object
            input_first_frame_image,  # hidden component used as ref when clearing points
            points_map,  # Image component where we add new tracking points
            video_frames_dir,  # Array where frames from video_in are deep stored
            scanned_frames,  # Scanned frames by EfficientTAM
            stored_inference_state,  # EfficientTAM inference state
            stored_frame_names,  #
            video_in_drawer,  # Accordion to hide uploaded video player
        ],
        queue=False,
    )

    video_in.change(
        fn=preprocess_video_in,
        inputs=[video_in],
        outputs=[
            first_frame_path,
            tracking_points,  # update Tracking Points in the gr.State([]) object
            trackings_input_label,  # update Tracking Labels in the gr.State([]) object
            input_first_frame_image,  # hidden component used as ref when clearing points
            points_map,  # Image component where we add new tracking points
            video_frames_dir,  # Array where frames from video_in are deep stored
            scanned_frames,  # Scanned frames by EfficientTAM
            stored_inference_state,  # EfficientTAM inference state
            stored_frame_names,  #
            video_in_drawer,  # Accordion to hide uploaded video player
        ],
        queue=False,
    )

    # triggered when we click on image to add new points
    points_map.select(
        fn=get_point,
        inputs=[
            point_type,  # "include" or "exclude"
            tracking_points,  # get tracking_points values
            trackings_input_label,  # get tracking label values
            input_first_frame_image,  # gr.State() first frame path
        ],
        outputs=[
            tracking_points,  # updated with new points
            trackings_input_label,  # updated with corresponding labels
            points_map,  # updated image with points
        ],
        queue=False,
    )

    # Clear every points clicked and added to the map
    clear_points_btn.click(
        fn=clear_points,
        inputs=input_first_frame_image,  # we get the untouched hidden image
        outputs=[
            first_frame_path,
            tracking_points,
            trackings_input_label,
            points_map,
        ],
        queue=False,
    )

    change_current.click(
        fn=switch_working_frame,
        inputs=[working_frame, scanned_frames, video_frames_dir],
        outputs=[
            tracking_points,
            trackings_input_label,
            input_first_frame_image,
            points_map,
        ],
        queue=False,
    )

    submit_btn.click(
        fn=get_mask_efficienttam_process,
        inputs=[
            stored_inference_state,
            input_first_frame_image,
            checkpoint,
            tracking_points,
            trackings_input_label,
            video_frames_dir,
            scanned_frames,
            working_frame,
            available_frames_to_check,
        ],
        outputs=[
            change_current,
            output_result,
            stored_frame_names,
            loaded_predictor,
            stored_inference_state,
            working_frame,
        ],
        concurrency_limit=10,
        queue=False,
    )

    reset_prpgt_brn.click(
        fn=reset_propagation,
        inputs=[first_frame_path, loaded_predictor, stored_inference_state],
        outputs=[
            points_map,
            tracking_points,
            trackings_input_label,
            output_propagated,
            stored_inference_state,
            output_result,
            available_frames_to_check,
            input_first_frame_image,
            working_frame,
            reset_prpgt_brn,
        ],
        queue=False,
    )

    propagate_btn.click(
        fn=update_ui,
        inputs=[vis_frame_type],
        outputs=[output_propagated, output_video],
        queue=False,
    ).then(
        fn=propagate_to_all,
        inputs=[
            tracking_points,
            video_in,
            checkpoint,
            stored_inference_state,
            stored_frame_names,
            video_frames_dir,
            vis_frame_type,
            available_frames_to_check,
            working_frame,
        ],
        outputs=[
            output_propagated,
            output_video,
            working_frame,
            available_frames_to_check,
            reset_prpgt_brn,
        ],
        concurrency_limit=10,
        queue=False,
    )

demo.queue()
demo.launch(share=True)
