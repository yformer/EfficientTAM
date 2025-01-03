#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Adapted from https://github.com/facebookresearch/sam2/blob/main/checkpoints/download_ckpts.sh

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Define the URLs for Efficient Track Anything checkpoints
EfficientTAM_BASE_URL="https://huggingface.co/yunyangx/efficient-track-anything/resolve/main"
efficienttam_s_url="${EfficientTAM_BASE_URL}/efficienttam_s.pt"
efficienttam_s_512x512_url="${EfficientTAM_BASE_URL}/efficienttam_s_512x512.pt"
efficienttam_s_1_url="${EfficientTAM_BASE_URL}/efficienttam_s_1.pt"
efficienttam_s_2_url="${EfficientTAM_BASE_URL}/efficienttam_s_2.pt"
efficienttam_ti_url="${EfficientTAM_BASE_URL}/efficienttam_ti.pt"
efficienttam_ti_512x512_url="${EfficientTAM_BASE_URL}/efficienttam_ti_512x512.pt"
efficienttam_ti_1_url="${EfficientTAM_BASE_URL}/efficienttam_ti_1.pt"
efficienttam_ti_2_url="${EfficientTAM_BASE_URL}/efficienttam_ti_2.pt"

# Efficient Track Anything checkpoints
echo "Downloading efficienttam_s.pt checkpoint..."
$CMD $efficienttam_s_url || { echo "Failed to download checkpoint from $efficienttam_s_url"; exit 1; }

echo "Downloading efficienttam_s_512x512.pt checkpoint..."
$CMD $efficienttam_s_512x512_url || { echo "Failed to download checkpoint from $efficienttam_s_512x512_url"; exit 1; }

echo "Downloading efficienttam_s_1.pt checkpoint..."
$CMD $efficienttam_s_1_url || { echo "Failed to download checkpoint from $efficienttam_s_1_url"; exit 1; }

echo "Downloading efficienttam_s_2.pt checkpoint..."
$CMD $efficienttam_s_2_url || { echo "Failed to download checkpoint from $efficienttam_s_2_url"; exit 1; }

echo "Downloading efficienttam_ti.pt checkpoint..."
$CMD $efficienttam_ti_url || { echo "Failed to download checkpoint from $efficienttam_ti_url"; exit 1; }

echo "Downloading efficienttam_ti_512x512.pt checkpoint..."
$CMD $efficienttam_ti_512x512_url || { echo "Failed to download checkpoint from $efficienttam_ti_512x512_url"; exit 1; }

echo "Downloading efficienttam_ti_1.pt checkpoint..."
$CMD $efficienttam_ti_1_url || { echo "Failed to download checkpoint from $efficienttam_ti_1_url"; exit 1; }

echo "Downloading efficienttam_ti_2.pt checkpoint..."
$CMD $efficienttam_ti_2_url || { echo "Failed to download checkpoint from $efficienttam_ti_2_url"; exit 1; }

echo "All efficient track anything checkpoints are downloaded successfully."
