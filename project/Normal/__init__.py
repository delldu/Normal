"""Surface Normal Estimation Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Thu 13 Jul 2023 01:55:56 PM CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
from PIL import Image

import torch
import todos
from torchvision import transforms as T
from .normal import NNET

import pdb


def create_model():
    """
    Create model
    """

    device = todos.model.get_device()
    model = NNET("BN")
    model = model.eval()
    model = model.to(device)
    print(f"Running model on {device} ...")

    return model, device


def get_model():
    """Load jit script model."""

    model, device = create_model()
    # print(model)

    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;
    # torch::jit::setTensorExprFuserEnabled(false);

    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/Normal.torch"):
        model.save("output/Normal.torch")

    return model, device


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_image = T.ToTensor()(image)
        input_image = input_image.unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_image)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([input_image, output_tensor], output_file)

    progress_bar.close()

    todos.model.reset_device()
