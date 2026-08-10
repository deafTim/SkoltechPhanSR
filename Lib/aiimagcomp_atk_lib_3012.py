############################################
# Algorithms for AI image compression attack
#
# ANH-HUY PHAN
############################################
import math
import os
import pickle
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import torch.nn.functional as F
from kornia.losses import PSNRLoss, SSIMLoss
from PIL import Image

# for binary mask
from skimage.morphology import dilation, erosion, opening, square
from torchvision import transforms

from torchmetrics.image import VisualInformationFidelity
vif = VisualInformationFidelity()
# # Move the VIF metric to the GPU
# vif = vif.to(device)

# Define a wrapper function for SSIMLoss
def ssim(input, target, window_size=11):
    ssim_loss = SSIMLoss(window_size=window_size)
    return ssim_loss(input, target)


def dists(input, target):
    dists_loss = piq.DISTS(reduction="none")
    return dists_loss(input, target)


def bpp_loss_0(output, num_pixels):
    bpp = (
        torch.log(output["likelihoods"]["y"]).sum()
        + torch.log(output["likelihoods"]["z"]).sum()
    ) / (-math.log(2) * num_pixels)
    return bpp


def bpp_loss(output, num_pixels):
    bpp = sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in output["likelihoods"].values()
    )
    return bpp


def export_results_to_image(
    perturbed_image, perturbed_output, noise_pattern, file_name, methodname
):
    file_no_extension = os.path.splitext(file_name)[0]

    perturbed_image_arr = (
        perturbed_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    )

    # ensure the values are in the range [0, 1]
    perturbed_image_arr = np.clip(perturbed_image_arr, 0, 1)

    # scale the values to [0, 255] and convert to uint8
    perturbed_image_arr = (perturbed_image_arr * 255).astype(np.uint8)

    # Create a PIL Image and save it
    im = Image.fromarray(perturbed_image_arr)
    file_name = file_no_extension + "_perturbed_" + methodname + ".png"
    im.save(file_name)

    # ----
    perturbed_output_arr = (
        perturbed_output.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    )

    # ensure the values are in the range [0, 1]
    perturbed_output_arr = np.clip(perturbed_output_arr, 0, 1)

    # scale the values to [0, 255] and convert to uint8
    perturbed_output_arr = (perturbed_output_arr * 255).astype(np.uint8)

    # Create a PIL Image and save it
    im = Image.fromarray(perturbed_output_arr)
    file_name = file_no_extension + "_decompress_" + methodname + ".png"

    im.save(file_name)

    # ----
    noise_pattern_arr = (
        noise_pattern.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    )

    # normalize shift the noise to 0-1
    minn = np.min(noise_pattern_arr.ravel())
    maxn = np.max(noise_pattern_arr.ravel())
    noise_pattern_arr = (noise_pattern_arr - minn) / (maxn - minn)

    # # ensure the values are in the range [0, 1]
    # noise_pattern_arr = np.clip(noise_pattern_arr, 0, 1)

    # scale the values to [0, 255] and convert to uint8
    noise_pattern_arr = (noise_pattern_arr * 255).astype(np.uint8)

    # Create a PIL Image and save it
    im = Image.fromarray(noise_pattern_arr)
    file_name = file_no_extension + "_noise_" + methodname + ".png"

    im.save(file_name)

    return


# -----for saving latents to file


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def savecompressed(compressfile, outnet, bitdepth, h, w):
    # with torch.no_grad():
    # outnet = net.compress(image)

    shape = outnet["shape"]

    with Path(compressfile).open("wb") as f:
        # write_uchars(f, codec.codec_header)
        # write original image size
        write_uints(f, (h, w))
        # write original bitdepth
        write_uchars(f, (bitdepth,))
        # write shape and number of encoded latents
        write_body(f, shape, outnet["strings"])

    size = filesize(compressfile)
    bpp = float(size) * 8 / (h * w)

    return bpp


def gaussian_kernel(size, sigma):
    # Create a vector of size 'size' with values from -size//2 to size//2
    x = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
    # Calculate the Gaussian distribution for each value in the vector
    g = torch.exp(-(x**2) / (2 * sigma**2))
    # Normalize the distribution so it sums to 1
    g /= g.sum()
    # Create a 2D Gaussian kernel from the outer product of the vector with itself
    return g.outer(g)




##___________________________________________#######


def maxdistortion_clamp(
    x,
    errbound=0.1,
    smoothfilter=None,
    losstype="psnr",
    l1_lambda=0,
    num_iterations=1000,
    model=None,
    device=None,
    mask=None,
    initial_noise=None,
    learningrate=0.1,
):
    """
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    """
    #
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    if learningrate is None:
        learningrate = 0.1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            noise_pattern.data.clamp_(-errbound, errbound)
            # noise_pattern2 = errbound * torch.tanh(noise_pattern)

            # Smooth the noise pattern
            if smoothfilter is None:
                smoothed_noise_pattern = noise_pattern * mask
            else:
                kernel_size = smoothfilter.shape[-1]
                smoothed_noise_pattern = F.conv2d(
                    noise_pattern * mask,
                    smoothfilter,
                    padding=kernel_size // 2,
                    groups=x.size(1),
                )

            # Apply current noise pattern
            perturbed_image = x + smoothed_noise_pattern

            # Forward pass through the model
            output = model(perturbed_image)

            # output['x_hat'] for the reconstructed image
            perturbed_output = output["x_hat"]

            if losstype == "psnr":
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I**2) / mse_loss)

                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR

            elif losstype == "ssim":
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)

                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM

            elif losstype == "dists":

                dists_perturbed = dists(x, perturbed_output)
                loss = -dists_perturbed

            if l1_lambda > 0:
                # L1-norm for sparsity
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm
            else:
                combined_loss = loss

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}")

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint(
            {
                "noise_pattern": noise_pattern.data,
                "smoothed_noise_pattern": smoothed_noise_pattern.data,
                "perturbed_image": perturbed_image.data,
                "perturbed_output": perturbed_output.data,
                "iteration": iteration,
            }
        )
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern



###_______________________________________####


def maxdistortion_tanh(
    x,
    errbound=0.1,
    smoothfilter=None,
    losstype="psnr",
    l1_lambda=0,
    num_iterations=1000,
    model=None,
    device=None,
    mask=None,
    initial_noise=None,
    learningrate=0.1,
):
    """
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    """
    #
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    if learningrate is None:
        learningrate = 0.1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            # noise_pattern.data.clamp_(-errbound, errbound)
            noise_pattern2 = errbound * torch.tanh(noise_pattern)

            # Smooth the noise pattern
            if smoothfilter is None:
                smoothed_noise_pattern = noise_pattern2 * mask
            else:
                kernel_size = smoothfilter.shape[-1]
                smoothed_noise_pattern = F.conv2d(
                    noise_pattern2 * mask,
                    smoothfilter,
                    padding=kernel_size // 2,
                    groups=x.size(1),
                )

            # Apply current noise pattern
            perturbed_image = x + smoothed_noise_pattern

            # Forward pass through the model
            output = model(perturbed_image)

            # output['x_hat'] for the reconstructed image
            perturbed_output = output["x_hat"]

            if losstype == "psnr":
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I**2) / mse_loss)

                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR

            elif losstype == "ssim":
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)

                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM

            elif losstype == "dists":

                dists_perturbed = dists(x, perturbed_output)
                loss = -dists_perturbed

            if l1_lambda > 0:
                # L1-norm for sparsity
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm
            else:
                combined_loss = loss

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}")

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint(
            {
                "noise_pattern": noise_pattern.data,
                "smoothed_noise_pattern": smoothed_noise_pattern.data,
                "perturbed_image": perturbed_image.data,
                "perturbed_output": perturbed_output.data,
                "iteration": iteration,
            }
        )
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


##___________________________________________#######


def maxdistortion(
    x,
    errbound,
    smoothfilter,
    losstype,
    l1_lambda,
    num_iterations,
    model,
    device=None,
    mask=None,
    initial_noise=None,
):
    """
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    """
    #
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=0.1)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            noise_pattern.data.clamp_(-errbound, errbound)

            # Smooth the noise pattern
            kernel_size = smoothfilter.shape[-1]
            smoothed_noise_pattern = F.conv2d(
                noise_pattern * mask,
                smoothfilter,
                padding=kernel_size // 2,
                groups=x.size(1),
            )

            # Apply current noise pattern
            perturbed_image = x + smoothed_noise_pattern

            # Forward pass through the model
            output = model(perturbed_image)

            # output['x_hat'] for the reconstructed image
            perturbed_output = output["x_hat"]

            if losstype == "psnr":
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I**2) / mse_loss)

                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR

            elif losstype == "ssim":
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)

                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM

            elif losstype == "dists":

                dists_perturbed = dists(x, perturbed_output)
                loss = -dists_perturbed

            if l1_lambda > 0:
                # L1-norm for sparsity
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm
            else:
                combined_loss = loss

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}")

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint(
            {
                "noise_pattern": noise_pattern.data,
                "smoothed_noise_pattern": smoothed_noise_pattern.data,
                "perturbed_image": perturbed_image.data,
                "perturbed_output": perturbed_output.data,
                "iteration": iteration,
            }
        )
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


###_______________________________________####


def maxbitrate_v0(
    x,
    errbound,
    smoothfilter,
    qualitymeasure,
    target_quality,
    quality_loss_lambda,
    l1_lambda,
    num_iterations,
    model,
    device=None,
    mask=None,
    initial_noise=None,
):
    # Attack the whole image with a noise pattern which
    # - maximizes the loss of the compression performance : maximize bpp
    # - preserves the PSNR of the decompressed image :   min |PSNR(f(x + n)) - PSNR(f(x))|
    # - Sparse and smooth perturbed noise
    """
    min_n -bpp(theta|n) + l_1 |PSNR(f(x + n)) - PSNR(f(x))| + ll_sparse ||x+n||
    s.t  |n_{i,kj}|<= sigma
    """
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # qualitymeasure: type of quality metric to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # quality_loss_lambda :  weight for quality loss regularizer
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=1)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Clamp the noise pattern values to ensure they stay within a valid range
        noise_pattern.data.clamp_(-errbound, errbound)

        # Smooth the noise pattern
        kernel_size = smoothfilter.shape[-1]
        smoothed_noise_pattern = F.conv2d(
            noise_pattern * mask,
            smoothfilter,
            padding=kernel_size // 2,
            groups=x.size(1),
        )

        # Apply current noise pattern
        perturbed_image = x + smoothed_noise_pattern

        # Forward pass through the model
        output = model(perturbed_image)

        # Assuming 'output' is a dictionary with key 'x_hat' for the reconstructed image
        perturbed_output = output["x_hat"]

        if qualitymeasure == "psnr":
            # Calculate MSE loss
            mse_loss = F.mse_loss(perturbed_output, x)

            # Calculate PSNR loss
            perturbed_quality = 10 * torch.log10((MAX_I**2) / mse_loss)

            # Compute the difference in PSNR between perturbed and target
            quality_loss = (perturbed_quality - target_quality).abs()

        elif qualitymeasure == "ssim":
            # maximize distortion = minimize 1-SSIM
            perturbed_quality = ssim(perturbed_output, x)

            quality_loss = (perturbed_quality - target_quality).abs()

        if l1_lambda > 0:
            # L1-norm for sparsity
            l1norm = smoothed_noise_pattern.abs().sum()

        else:
            l1norm = 0

        # Compute the bpp loss
        bpploss = bpp_loss(output, num_pixels)

        # quality_loss_lambda = 0.1

        # Combine the losses
        # combined_loss = -bpploss + quality_loss_lambda * quality_loss + l1_lambda * l1norm

        combined_loss = (
            -10 * torch.log10(bpploss)
            + quality_loss_lambda * torch.log10(quality_loss)
            + l1_lambda * l1norm
        )

        # Perform gradient descent
        combined_loss.backward()
        optimizer.step()

        # Print the loss every 100 iterations
        if iteration % 100 == 0:

            print(
                f"Iteration {iteration} | {qualitymeasure}: {perturbed_quality} - Lost {quality_loss} | BPP {bpploss} |  Loss {combined_loss}"
            )

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


def noise_to_mask(smoothed_noise_pattern):
    # Anh-Huy Phan 
    #
    # Define a mask based on the sparse noise pattern
    ell2_noise = torch.sqrt(torch.sum(smoothed_noise_pattern**2, dim=1))

    mask_noise = ell2_noise > torch.max(ell2_noise) * 1e-1
    mask_noise = mask_noise.squeeze()

    mask_noise = erosion(mask_noise.cpu().detach().numpy(), square(5))
    mask_noise = opening(mask_noise, square(5))
    mask_noise = dilation(mask_noise, square(5))
    mask_noise = dilation(mask_noise, square(5))

    # plt.imshow(mask_noise)

    # Mask 3D
    mask = torch.zeros_like(smoothed_noise_pattern)
    nnz_ix = np.where(mask_noise == 1)
    mask[:, :, nnz_ix[0], nnz_ix[1]] = 1

    return mask, mask_noise


# Function to save state
def save_checkpoint(state, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(state, f)


# Function to load state
def load_checkpoint(filename="checkpoint.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    return None


def maxbitrate(
    x,
    errbound,
    smoothfilter,
    qualitymeasure,
    target_quality,
    quality_loss_lambda,
    l1_lambda,
    num_iterations,
    model,
    device=None,
    mask=None,
    initial_noise=None,
):
    # Attack the whole image with a noise pattern which
    # - maximizes the loss of the compression performance : maximize bpp
    # - preserves the PSNR of the decompressed image :   min |PSNR(f(x + n)) - PSNR(f(x))|
    # - Sparse and smooth perturbed noise
    """
    min_n -bpp(theta|n) + l_1 |PSNR(f(x + n)) - PSNR(f(x))| + ll_sparse ||x+n||
    s.t  |n_{i,kj}|<= sigma
    """
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # qualitymeasure: type of quality metric to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # quality_loss_lambda :  weight for quality loss regularizer
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=1)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            noise_pattern.data.clamp_(-errbound, errbound)

            # Smooth the noise pattern
            kernel_size = smoothfilter.shape[-1]
            smoothed_noise_pattern = F.conv2d(
                noise_pattern * mask,
                smoothfilter,
                padding=kernel_size // 2,
                groups=x.size(1),
            )

            # Apply current noise pattern
            perturbed_image = x + smoothed_noise_pattern

            # Forward pass through the model
            output = model(perturbed_image)

            # Assuming 'output' is a dictionary with key 'x_hat' for the reconstructed image
            perturbed_output = output["x_hat"]

            if qualitymeasure == "psnr":
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                perturbed_quality = 10 * torch.log10((MAX_I**2) / mse_loss)

                # Compute the difference in PSNR between perturbed and target
                quality_loss = (perturbed_quality - target_quality).abs()

            elif qualitymeasure == "ssim":
                # maximize distortion = minimize 1-SSIM
                perturbed_quality = ssim(perturbed_output, x)

                quality_loss = (perturbed_quality - target_quality).abs()

            if l1_lambda > 0:
                # L1-norm for sparsity
                l1norm = smoothed_noise_pattern.abs().sum()

            else:
                l1norm = 0

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # quality_loss_lambda = 0.1

            # Combine the losses
            # combined_loss = -bpploss + quality_loss_lambda * quality_loss + l1_lambda * l1norm

            combined_loss = (
                -10 * torch.log10(bpploss)
                + quality_loss_lambda * torch.log10(quality_loss)
                + l1_lambda * l1norm
            )

            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:

                print(
                    f"Iteration {iteration} | {qualitymeasure}: {perturbed_quality} - Lost {quality_loss} | BPP {bpploss} |  Loss {combined_loss}"
                )

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint(
            {
                "noise_pattern": noise_pattern.data,
                "smoothed_noise_pattern": smoothed_noise_pattern.data,
                "perturbed_image": perturbed_image.data,
                "perturbed_output": perturbed_output.data,
                "iteration": iteration,
            }
        )
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


import copy

# Entropy filtering
from scipy.stats import entropy
from skimage import color, io, segmentation
from skimage.filters.rank import entropy as entropy_filter
from skimage.measure import regionprops
from skimage.morphology import disk

from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float


###_______________________________________###
# For entropy filter ________________________


def noisemask_maxentropy_superpixel(
    img_rgb, n_segments=400, sigma=4, n_topentropysegments=5, verbose=False
):
    # Return the mask for the selected region of the images (superpixels) which have the highest entropy
    #
    # Anh-Huy Phan 
    #
    # SLIC super-pixel
    segments = slic(img_rgb, n_segments=n_segments, sigma=sigma)
    imsize_ = img_rgb.shape
    #
    regions = regionprops(segments)

    # Convert to grayscale as entropy is typically computed on grayscale images
    gray_img = color.rgb2gray(img_rgb)

    # Compute histogram for each superpixel
    histograms = []
    for i, segVal in enumerate(np.unique(segments)):
        # Mask the superpixel
        mask = np.zeros(gray_img.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255

        # Compute the histogram
        hist, _ = np.histogram(gray_img[mask > 0], bins=256, range=(0, 1))
        histograms.append(hist)

    # Compute the entropy for each histogram
    entropies = [entropy(hist) for hist in histograms]

    # Select superpixels with high entropy
    # threshold = np.mean(entropies)
    # low_entropy_superpixels = [i for i, e in enumerate(entropies) if e < threshold]
    superpixel_order = np.argsort(entropies)

    # select superpixel with highest entropy
    # superpixel_order: index list of superpixels ordered in the ascending of their entropy
    top_entropy_superpixels = superpixel_order[-n_topentropysegments:]

    # Binary mask for the selected superpixel
    # mask = torch.zeros_like([1 img_rgbx)
    mask = torch.zeros([1, imsize_[2], imsize_[0], imsize_[1]])
    # Set ones at the specified coordinates in the mask
    for segVal in top_entropy_superpixels:
        coord = regions[segVal]["coords"]
        mask[:, :, coord[:, 0], coord[:, 1]] = 1

    # Initialize an array to hold the maximum entropy value for each superpixel
    mean_entropy_segment_img = np.zeros_like(gray_img)
    # Iterate over each unique segment label
    for i, segVal in enumerate(np.unique(segments)):
        # Create a mask for the current segment
        mask_k = segments == segVal

        # Assign this mean value to the corresponding pixels in the output array
        mean_entropy_segment_img[mask_k] = entropies[i]

    if verbose == True:
        # display maps of superpixels
        plt.figure(1)

        # Overlay the segment boundaries in yellow
        minn = np.min(mean_entropy_segment_img.reshape(-1))
        maxn = np.max(mean_entropy_segment_img.reshape(-1))

        mm = (mean_entropy_segment_img - minn) / (maxn - minn)
        plt.imshow(mm)
        marked = mark_boundaries(mm, segments, color=(1, 1, 0))
        # Display the boundaries on top of the colored image
        plt.imshow(marked, alpha=0.2, cmap="gray", interpolation="none")

        # # Show the plot
        plt.show()

        # display maps of superpixels
        plt.figure(2)
        plt.imshow(mask.squeeze().cpu().detach().numpy().transpose(1, 2, 0))

    return mask, superpixel_order, regions, segments, mean_entropy_segment_img, entropies



def noisemask_maxentropy_superpixel_new(img_rgb, n_segments=400, sigma=4, n_topentropysegments=5, verbose=False):
    # Return the mask for the selected region of the images (superpixels) which have the highest entropy

    # SLIC super-pixel
    segments = slic(img_rgb, n_segments=n_segments, sigma=sigma)
    imsize_ = img_rgb.shape
    
    # Convert to grayscale as entropy is typically computed on grayscale images
    gray_img = color.rgb2gray(img_rgb)

    # Compute histogram for each superpixel
    histograms = []
    for (i, segVal) in enumerate(np.unique(segments)):
        # Mask the superpixel
        mask = np.zeros(gray_img.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255
    
        # Compute the histogram
        hist, _ = np.histogram(gray_img[mask > 0], bins=256, range=(0, 1))
        histograms.append(hist)

    # Calculate the fraction of histogram bins where the value is greater than 0.75
    fractions = [(hist > 0.75).sum() / len(hist) for hist in histograms]

    # Select superpixels with high fractions
    superpixel_order = np.argsort(fractions)

    # Select superpixels with highest fractions
    top_fraction_superpixels = superpixel_order[-n_topentropysegments:]

    # Binary mask for the selected superpixel
    mask = torch.zeros([1, imsize_[2], imsize_[0], imsize_[1]])
    regions = regionprops(segments)
    # Set ones at the specified coordinates in the mask
    for segVal in top_fraction_superpixels:
        coord = regions[segVal]["coords"]
        mask[:, :, coord[:, 0], coord[:, 1]] = 1

    # Initialize an array to hold the fraction value for each superpixel
    fraction_segment_img = np.zeros_like(gray_img)
    # Iterate over each unique segment label
    for (i, segVal) in enumerate(np.unique(segments)):
        # Create a mask for the current segment
        mask_k = segments == segVal
         
        # Assign this mean value to the corresponding pixels in the output array
        fraction_segment_img[mask_k] = fractions[i]

    if verbose:
        # display maps of superpixels 
        plt.figure(1)
       
        # Overlay the segment boundaries in yellow
        minn = np.min(fraction_segment_img.reshape(-1))
        maxn = np.max(fraction_segment_img.reshape(-1))
        
        mm = (fraction_segment_img - minn) / (maxn - minn)
        plt.imshow(mm)    
        marked = mark_boundaries(mm, segments, color=(1, 1, 0))
        # Display the boundaries on top of the colored image
        plt.imshow(marked, alpha=0.2, cmap='gray', interpolation='none')
        
        # Show the plot
        plt.show()
    
        # display maps of superpixels 
        plt.figure(2)
        plt.imshow(mask.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
        
    return mask, superpixel_order, regions, segments, fraction_segment_img


def noisemask_entropyfilter(
    img_rgb, n_segments=400, sigma=4, n_topentropysegments=5, verbose=False
):
    # Return the mask for the selected region of the images (superpixels) which have the highest entropy
    #
    # Anh-Huy Phan 
    #
    imsize_ = img_rgb.shape
    gray_img = color.rgb2gray(img_rgb)

    # Entropy image
    entr_img = entropy_filter(gray_img, disk(5))

    # Superpixel
    segments = slic(img_rgb, n_segments=n_segments, sigma=sigma)
    regions = regionprops(segments)

    # Initialize an array to hold the maximum entropy value for each superpixel
    mean_entropy_segment_img = np.zeros_like(entr_img)
    mean_entropy_segment = np.zeros(len(np.unique(segments)))
    # Iterate over each unique segment label
    for k in np.unique(segments):
        # Create a mask for the current segment
        mask_k = segments == k

        # Get all entropy values in the current segment
        segment_entropy_values = entr_img[mask_k]

        # Calculate the mean of the top 20 entropy values
        mean_entropy = np.mean(segment_entropy_values)

        # Assign this mean value to the corresponding pixels in the output array
        mean_entropy_segment_img[mask_k] = mean_entropy
        mean_entropy_segment[k - 1] = mean_entropy

    # low_entropy_superpixels = [i for i, e in enumerate(entropies) if e < threshold]
    superpixel_order = np.argsort(mean_entropy_segment)

    # superpixel_order: index list of superpixels ordered in the ascending of their entropy
    top_entropy_superpixels = superpixel_order[-n_topentropysegments:]

    # Binary mask for the selected superpixel
    mask = torch.zeros([1, imsize_[2], imsize_[0], imsize_[1]])
    # Set ones at the specified coordinates in the mask
    for segVal in top_entropy_superpixels:
        coord = regions[segVal]["coords"]
        mask[:, :, coord[:, 0], coord[:, 1]] = 1

    if verbose == True:
        # display maps of superpixels
        plt.figure(1)

        # Overlay the segment boundaries in yellow
        minn = np.min(mean_entropy_segment_img.reshape(-1))
        maxn = np.max(mean_entropy_segment_img.reshape(-1))

        mm = (mean_entropy_segment_img - minn) / (maxn - minn)
        plt.imshow(mm)
        marked = mark_boundaries(mm, segments, color=(1, 1, 0))
        # Display the boundaries on top of the colored image
        plt.imshow(marked, alpha=0.2, cmap="gray", interpolation="none")

        # # Show the plot
        plt.show()

        # display maps of superpixels
        plt.figure(2)
        plt.imshow(mask.squeeze().cpu().detach().numpy().transpose(1, 2, 0))

    return mask, superpixel_order, regions, segments, mean_entropy_segment_img


def noisemask_fractionfilter_new(img_rgb, n_segments=400, sigma=4, n_topsegments=5, verbose=False):
    # Return the mask for the selected region of the images (superpixels) which have the highest fraction of high histogram values
    # Anh-Huy Phan 2024
    
    imsize_ = img_rgb.shape
    gray_img = color.rgb2gray(img_rgb)

    # Entropy image (using entropy filter)
    entr_img = entropy_filter(gray_img, disk(5))
    
    # Superpixel segmentation
    segments = slic(img_rgb, n_segments=n_segments, sigma=sigma)
    regions = regionprops(segments)

    # Initialize an array to hold the fraction value for each superpixel
    fraction_segment_img = np.zeros_like(entr_img)
    fraction_segment = np.zeros(len(np.unique(segments)))
    
    # Iterate over each unique segment label
    for k in np.unique(segments):
        # Create a mask for the current segment
        mask_k = segments == k
    
        # Get all entropy values in the current segment
        segment_entropy_values = entr_img[mask_k]
        
        # Compute the histogram
        hist, _ = np.histogram(segment_entropy_values, bins=256, range=(0, np.max(segment_entropy_values)))
        
        # Calculate the fraction of histogram bins where the value is greater than 0.75
        fraction = (hist > 0.75).sum() / len(hist)
     
        # Assign this fraction value to the corresponding pixels in the output array
        fraction_segment_img[mask_k] = fraction
        fraction_segment[k - 1] = fraction
    
    # Order superpixels by the fraction value
    superpixel_order = np.argsort(fraction_segment)

    # Select superpixels with highest fractions
    top_fraction_superpixels = superpixel_order[-n_topsegments:] 
    
    # Binary mask for the selected superpixels
    mask = torch.zeros([1, imsize_[2], imsize_[0], imsize_[1]])
    # Set ones at the specified coordinates in the mask
    for segVal in top_fraction_superpixels:
        coord = regions[segVal]["coords"]
        mask[:, :, coord[:, 0], coord[:, 1]] = 1

    if verbose:
        # Display maps of superpixels
        plt.figure(1)
       
        # Overlay the segment boundaries in yellow
        minn = np.min(fraction_segment_img.reshape(-1))
        maxn = np.max(fraction_segment_img.reshape(-1))
        
        mm = (fraction_segment_img - minn) / (maxn - minn)
        plt.imshow(mm)    
        marked = mark_boundaries(mm, segments, color=(1, 1, 0))
        # Display the boundaries on top of the colored image
        plt.imshow(marked, alpha=0.2, cmap='gray', interpolation='none')
        
        # Show the plot
        plt.show()
    
        # Display maps of superpixels
        plt.figure(2)
        plt.imshow(mask.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
    
    return mask, superpixel_order, regions, segments, fraction_segment_img

from scipy.ndimage import gaussian_filter

def noisemask_maxintensity_superpixel(img_rgb, n_segments=400, gauss_sigma=4, n_topsegments=5, verbose=False):
    # Return the mask for the selected region of the images (superpixels) which have the highest intensities
    
    # SLIC super-pixel segmentation
    segments = slic(img_rgb, n_segments=n_segments, sigma=4)
    imsize_ = img_rgb.shape
    regions = regionprops(segments)

    # Convert to grayscale
    #gray_img = color.rgb2gray(img_rgb)

    # Apply Gaussian filter to the grayscale image
    #filtered_img = gaussian_filter(gray_img, sigma=gauss_sigma)

    # Compute mean intensity for each superpixel
    mean_intensities = []
    mask3d = np.zeros_like(img_rgb)
    for segVal in np.unique(segments):
        # Mask the superpixel
        mask = segments == segVal  
        mask3d[:,:,0] = (mask)
        mask3d[:,:,1] = (mask)
        mask3d[:,:,2] = (mask)
        

        # Compute the mean intensity for the superpixel
        # plt.imshow(img_rgb*mask3d)
        # plt.pause(1)
        filtered_img = gaussian_filter(img_rgb*mask3d, sigma=gauss_sigma) 
        #mean_intensity = np.mean(img_rgb[mask])
        mean_intensity = np.mean(filtered_img)
        mean_intensities.append(mean_intensity)

    # Order superpixels by mean intensity
    superpixel_order = np.argsort(mean_intensities)

    # Select superpixels with highest mean intensities
    top_intensity_superpixels = superpixel_order[-n_topsegments:] 
    
    # Binary mask for the selected superpixel
    mask = torch.zeros([1, imsize_[2], imsize_[0], imsize_[1]])
    # Set ones at the specified coordinates in the mask
    for segVal in top_intensity_superpixels:
        coord = regions[segVal]["coords"]
        mask[:, :, coord[:, 0], coord[:, 1]] = 1

    # Initialize an array to hold the mean intensity value for each superpixel
    mean_intensity_segment_img = np.zeros_like(img_rgb)
    # Iterate over each unique segment label
    for (i, segVal) in enumerate(np.unique(segments)):
        # Create a mask for the current segment
        mask_k = segments == segVal
        # Assign this mean value to the corresponding pixels in the output array
        mean_intensity_segment_img[mask_k] = mean_intensities[i]

    if verbose:
        # Display maps of superpixels 
        plt.figure(1)
       
        # Overlay the segment boundaries in yellow
        minn = np.min(mean_intensity_segment_img.reshape(-1))
        maxn = np.max(mean_intensity_segment_img.reshape(-1))
        
        mm = (mean_intensity_segment_img - minn) / (maxn - minn)
        plt.imshow(mm)    
        marked = mark_boundaries(mm, segments, color=(1, 1, 0))
        # Display the boundaries on top of the colored image
        plt.imshow(marked, alpha=0.2, cmap='gray', interpolation='none')
        
        # Show the plot
        plt.show()
    
        # Display maps of superpixels 
        plt.figure(2)
        plt.imshow(mask.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
        
    return mask, superpixel_order, regions, segments, mean_intensity_segment_img

###_______________________________________###
# Calculate baseline peformance for the original output
def eval_perf(model, net, x, img_path):
    # Anh-Huy Phan 
    #
    MAX_I = 1
    num_pixels = x.shape[2] * x.shape[3]
    # model and net are the same , but net is on cpu and for compression
    with torch.no_grad():
        original_output = model(x)
        mse_loss_original = F.mse_loss(original_output["x_hat"].clamp_(0, 1), x)
        target_psnr = 10 * torch.log10((MAX_I**2) / mse_loss_original)

        # Compute bpp loss (to be maximized, hence the negative sign)
        baseline_bpp = (
            torch.log(original_output["likelihoods"]["y"]).sum()
            + torch.log(original_output["likelihoods"]["z"]).sum()
        ) / (-math.log(2) * num_pixels)

        target_ssim = ssim(original_output["x_hat"].clamp_(0, 1), x)

        original_output = net.compress(x.to("cpu"))

    # Generate a random number
    unique_id = np.random.randint(1000, 9999)

    # Create the new filename with 'compress' and the unique_id
    compressfile = os.path.splitext(img_path)[0] + "compress" + str(unique_id)

    bitdepth = 8
    h, w = x.size(2), x.size(3)
    baseline_true_bpp = savecompressed(compressfile, original_output, bitdepth, h, w)

    result = {
        "PSNR": target_psnr.cpu().detach().numpy(),
        "Bpp": baseline_bpp.cpu().detach().numpy(),
        "Bpp(fsize)": baseline_true_bpp,
        "SSIM": target_ssim.cpu().detach().numpy(),
    }
    return result

#########

###_______________________________________###
# Calculate baseline peformance for the original output
def eval_perf_full(model, net, perturbed_image, x,img_path):
    # Anh-Huy Phan 
    #
    global vif  # Declare vif as global to use the global variable

#     vif_module = 'VisualInformationFidelity'
 
#     if vif_module not in sys.modules:
    from torchmetrics.image import VisualInformationFidelity
    vif = VisualInformationFidelity()

    MAX_I = 1
    num_pixels = perturbed_image.shape[2] * perturbed_image.shape[3]
    # model and net are the same , but net is on cpu and for compression
    with torch.no_grad():
        perturbed_output = model(perturbed_image)
        perturbed_output_image = perturbed_output["x_hat"].clamp_(0, 1)
        
        mse_loss_original = F.mse_loss(perturbed_output_image, perturbed_image)
        target_psnr = 10 * torch.log10((MAX_I**2) / mse_loss_original)

        # Compute bpp loss (to be maximized, hence the negative sign)
        baseline_bpp = (
            torch.log(perturbed_output["likelihoods"]["y"]).sum()
            + torch.log(perturbed_output["likelihoods"]["z"]).sum()
        ) / (-math.log(2) * num_pixels)

        target_ssim = ssim(perturbed_output["x_hat"].clamp_(0, 1), x)

        original_output = net.compress(perturbed_image.to("cpu"))

    # Generate a random number
    unique_id = np.random.randint(1000, 9999)

    # Create the new filename with 'compress' and the unique_id
    compressfile = os.path.splitext(img_path)[0] + "compress" + str(unique_id)

    bitdepth = 8
    h, w = perturbed_image.size(2), perturbed_image.size(3)
    baseline_true_bpp = savecompressed(compressfile, original_output, bitdepth, h, w)
    
    # VIF
    device = x.device
    vif = vif.to(device)
    vif_score_in= vif(x, perturbed_image) 
    vif_score_out = vif(x, perturbed_output_image)
    mse_perturbed_original = F.mse_loss(x, perturbed_image)
    psnr_in = 10 * torch.log10((MAX_I**2) / mse_perturbed_original)

    result = {
        "PSNR(ai,ao)": target_psnr.cpu().detach().numpy(),
        "PSNR(ai,oi)": psnr_in.cpu().detach().numpy(),
        "Bpp": baseline_bpp.cpu().detach().numpy(),
        "Bpp(fsize)": baseline_true_bpp,
        "SSIM(ao)": target_ssim.cpu().detach().numpy(),
        "VIF(ai,oi)": vif_score_in.cpu().detach().numpy(),
        "VIF(ao,oi)": vif_score_out.cpu().detach().numpy(),
    }
    return result


###_________________________________####


def vis_results(perturbed_image, perturbed_output, smoothed_noise_pattern):
    # Visualize the compression result
    #
    # Anh-Huy Phan 
    
    plt.figure(1)
    plt.imshow(perturbed_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.show
    plt.figure(2)
    plt.imshow(perturbed_output.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.show
    plt.figure(3)
    minn = np.min(
        smoothed_noise_pattern.squeeze()
        .cpu()
        .detach()
        .numpy()
        .transpose(1, 2, 0)
        .ravel()
    )
    maxn = np.max(
        smoothed_noise_pattern.squeeze()
        .cpu()
        .detach()
        .numpy()
        .transpose(1, 2, 0)
        .ravel()
    )
    plt.imshow(
        (
            smoothed_noise_pattern.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
            - minn
        )
        / (maxn - minn)
    )
    plt.axis("off")
    plt.show

    return


import pandas as pd


# Function to check if two dictionaries have the same structure
def have_same_structure(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if type(dict1[key]) != type(dict2[key]):
            return False
    return True


# Function to collect performance data into a DataFrame
def collect_perf(all_vars, template_var_name):
    # Find variables with the same structure as 'template'
    # Anh-Huy Phan 
    similar_structures = [
        name
        for name, var in all_vars.items()
        if isinstance(var, dict)
        and have_same_structure(all_vars[template_var_name], var)
    ]

    # Create a DataFrame from the dictionaries
    data = {"Method": [], "PSNR": [], "Bpp": [], "Bpp(fsize)": [], "SSIM": []}

    # Populate the DataFrame
    for var_name in similar_structures:
        var = all_vars[var_name]
        data["Method"].append(var_name)
        data["PSNR"].append(var["PSNR"])
        data["Bpp"].append(var["Bpp"])
        data["Bpp(fsize)"].append(var["Bpp(fsize)"])
        data["SSIM"].append(var["SSIM"])

    # Convert the dictionary to a DataFrame and sort it
    df = pd.DataFrame(data)
    df = df.sort_values("Bpp")

    return df


# # Usage example:
# # Assuming 'baseline_' is the name of one of the dictionaries  
# all_vars = vars().copy()
# df = collect_perf(all_vars, 'baseline_')
# print(df)


def have_same_structure2(dict1, dict2):
    # Placeholder for the actual implementation
    return set(dict1.keys()) == set(dict2.keys())


# Improved Function to collect performance data into a DataFrame
def collect_perf2(all_vars, template_var_name):
    """
    Collects performance data from variables with the same structure as the template variable
    and returns a sorted DataFrame.

    Parameters:
    all_vars (dict): Dictionary containing all variables.
    template_var_name (str): Name of the template variable.

    Returns:
    pd.DataFrame: DataFrame containing the performance data.
    # Anh-Huy Phan 
    """
    # Ensure the template variable exists and is a dictionary
    if template_var_name not in all_vars or not isinstance(
        all_vars[template_var_name], dict
    ):
        raise ValueError(
            f"Template variable '{template_var_name}' is not found or is not a dictionary."
        )

    # Initialize the data dictionary with keys from the template variable
    data = {key: [] for key in all_vars[template_var_name].keys()}
    data["Method"] = []  # Add 'Method' key for storing variable names

    similar_structures = [
        name
        for name, var in all_vars.items()
        if isinstance(var, dict)
        and have_same_structure2(all_vars[template_var_name], var)
    ]

    # Populate the DataFrame
    for var_name in similar_structures:
        var = all_vars[var_name]

        data["Method"].append(var_name)
        for key, value in var.items():

            # Check if the value is a tensor and if it's on the GPU
            if torch.is_tensor(value) and value.is_cuda:
                # Move the tensor to the CPU
                data[key].append(value.cpu().detach().numpy())
            else:
                data[key].append(value)

    df = pd.DataFrame(data)

    columns = ["Method"] + [col for col in df.columns if col != "Method"]
    df = df.reindex(columns=columns)
    df = df.sort_values("Method")

    return df



# Define a function to check if a name is "strange"
def is_strange(name):
    import re
    # Check if the name starts with an underscore or contains only numbers
    return bool(re.match(r'^_+$', name)) or bool(re.match(r'^_\d+$', name))

# Improved Function to collect performance data into a DataFrame
def collect_perf_3(all_vars, template_var_name):
    """
    Collects performance data from variables with the same structure as the template variable
    and returns a sorted DataFrame.

    Parameters:
    all_vars (dict): Dictionary containing all variables.
    template_var_name (str): Name of the template variable.

    Returns:
    pd.DataFrame: DataFrame containing the performance data.
    # Anh-Huy Phan 
    """
    
    # Find variables with the same structure as 'template'
    similar_structures = [
        name
        for name, var in all_vars.items()
        if isinstance(var, dict)
        and have_same_structure(all_vars[template_var_name], var)
    ]
    similar_structures

    
 
    # Filter out strange names
    similar_structures = [name for name in similar_structures if not is_strange(name)]
    similar_structures

    # all_vars[similar_structures[1]]
    import pandas as pd

    # Extract the variables from all_vars
    extracted_vars = {name: all_vars[name] for name in similar_structures}

    df = pd.DataFrame(extracted_vars)
    df = df.transpose()
   
    df = df.rename_axis(columns="Method")
        
    return df, extracted_vars


##___________________________________________#######


def maxdistortion_multiplicativenoise_tanh(
    x,
    errbound=0.1,
    smoothfilter=None,
    losstype="psnr",
    l1_lambda=0,
    num_iterations=1000,
    model=None,
    device=None,
    mask=None,
    initial_noise=None,
    learningrate=1,
):
    """
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    """
    #
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    beta = 1
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Clamp the noise pattern values to ensure they stay within a valid range
        # #noise_pattern.data.clamp_(-errbound, errbound)

        # # Smooth the noise pattern
        # kernel_size = smoothfilter.shape[-1]
        # smoothed_noise_pattern = F.conv2d(noise_pattern * mask, smoothfilter, padding=kernel_size // 2, groups=x.size(1))

        # noise_pattern.data.clamp_(-errbound, errbound)
        noise_pattern2 = errbound * torch.tanh(noise_pattern * beta)

        # noise_pattern2 = errbound*(2/(1+torch.exp(-5*noise_pattern))-1)

        # Smooth the noise pattern
        if smoothfilter is None:
            smoothed_noise_pattern = noise_pattern2 * mask
        else:
            kernel_size = smoothfilter.shape[-1]
            smoothed_noise_pattern = F.conv2d(
                noise_pattern2 * mask,
                smoothfilter,
                padding=kernel_size // 2,
                groups=x.size(1),
            )

        # Apply current noise pattern
        perturbed_image = x * (1 + smoothed_noise_pattern)

        # Forward pass through the model
        output = model(perturbed_image)

        # output['x_hat'] for the reconstructed image
        perturbed_output = output["x_hat"]

        if losstype == "psnr":
            # Calculate MSE loss
            mse_loss = F.mse_loss(perturbed_output, x)

            # Calculate PSNR loss
            psnr_loss = 10 * torch.log10((MAX_I**2) / mse_loss)

            # maximize distortion = minimize PSNR
            loss = psnr_loss  # Negative sign because we want to maximize PSNR

        elif losstype == "ssim":
            # maximize distortion = minimize 1-SSIM
            ssim_perturbed = ssim(perturbed_output, x)

            loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM

        elif losstype == "dists":

            dists_perturbed = dists(x, perturbed_output)
            loss = -dists_perturbed

        if l1_lambda > 0:
            # L1-norm for sparsity
            l1norm = smoothed_noise_pattern.abs().sum()
            # Combine the loss with sparse L1 norm
            combined_loss = loss + l1_lambda * l1norm
        else:
            combined_loss = loss

        # Compute the bpp loss
        bpploss = bpp_loss(output, num_pixels)

        # Perform gradient descent
        combined_loss.backward()
        optimizer.step()

        # Print the loss every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}")

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


def maxbitrate_multiplicativenoise(
    x,
    errbound,
    smoothfilter,
    qualitymeasure,
    target_quality,
    quality_loss_lambda,
    l1_lambda,
    num_iterations=1000,
    model=None,
    device=None,
    mask=None,
    initial_noise=None,
    learningrate=1,
):
    # Attack the whole image with a noise pattern which
    # - maximizes the loss of the compression performance : maximize bpp
    # - preserves the PSNR of the decompressed image :   min |PSNR(f(x + n)) - PSNR(f(x))|
    # - Sparse and smooth perturbed noise
    """
    min_n -bpp(theta|n) + l_1 |PSNR(f(x + n)) - PSNR(f(x))| + ll_sparse ||x+n||
    s.t  |n_{i,kj}|<= sigma
    """
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # qualitymeasure: type of quality metric to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # quality_loss_lambda :  weight for quality loss regularizer
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            noise_pattern.data.clamp_(-errbound, errbound)

            # Smooth the noise pattern
            kernel_size = smoothfilter.shape[-1]
            smoothed_noise_pattern = F.conv2d(
                noise_pattern * mask,
                smoothfilter,
                padding=kernel_size // 2,
                groups=x.size(1),
            )

            # Apply current noise pattern
            perturbed_image = x * (1 + smoothed_noise_pattern)

            # Forward pass through the model
            output = model(perturbed_image)

            # Assuming 'output' is a dictionary with key 'x_hat' for the reconstructed image
            perturbed_output = output["x_hat"]

            if qualitymeasure == "psnr":
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                perturbed_quality = 10 * torch.log10((MAX_I**2) / mse_loss)

                # Compute the difference in PSNR between perturbed and target
                quality_loss = (perturbed_quality - target_quality).abs()

            elif qualitymeasure == "ssim":
                # maximize distortion = minimize 1-SSIM
                perturbed_quality = ssim(perturbed_output, x)

                quality_loss = (perturbed_quality - target_quality).abs()

            if l1_lambda > 0:
                # L1-norm for sparsity
                l1norm = smoothed_noise_pattern.abs().sum()

            else:
                l1norm = 0

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # quality_loss_lambda = 0.1

            # Combine the losses
            # combined_loss = -bpploss + quality_loss_lambda * quality_loss + l1_lambda * l1norm

            combined_loss = (
                -10 * torch.log10(bpploss)
                + quality_loss_lambda * torch.log10(quality_loss)
                + l1_lambda * l1norm
            )

            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:

                print(
                    f"Iteration {iteration} | {qualitymeasure}: {perturbed_quality} - Lost {quality_loss} | BPP {bpploss} |  Loss {combined_loss}"
                )

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint(
            {
                "noise_pattern": noise_pattern.data,
                "smoothed_noise_pattern": smoothed_noise_pattern.data,
                "perturbed_image": perturbed_image.data,
                "perturbed_output": perturbed_output.data,
                "iteration": iteration,
            }
        )
        print("Interrupted, checkpoint saved.")

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern



###----------------------------------####
def maxbitrate_tanh(x, errbound=0.1, smoothfilter=None, qualitymeasure='psnr', target_quality=None,quality_loss_lambda=0.1,l1_lambda=0, num_iterations=1000, model=None, device=None, mask=None,initial_noise=None,learningrate=1):
    
    # Attack the whole image with a noise pattern which 
    # - maximizes the loss of the compression performance : maximize bpp 
    # - preserves the PSNR of the decompressed image :   min |PSNR(f(x + n)) - PSNR(f(x))|
    # - Sparse and smooth perturbed noise
    '''
        min_n -bpp(theta|n) + l_1 |PSNR(f(x + n)) - PSNR(f(x))| + ll_sparse ||x+n||
        s.t  |n_{i,kj}|<= sigma
    '''
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # qualitymeasure: type of quality metric to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # quality_loss_lambda :  weight for quality loss regularizer 
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1
        
    if learningrate is None:
        learningrate = 1

    if target_quality is None:
        target_quality = 0

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(device)
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    kappa = 5
    
    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            #noise_pattern.data.clamp_(-errbound, errbound)
            noise_pattern2 = errbound*torch.tanh(noise_pattern*kappa)
 
            # Smooth the noise pattern
            if smoothfilter is None:
                smoothed_noise_pattern = noise_pattern2 * mask
            else:
                kernel_size = smoothfilter.shape[-1]
                smoothed_noise_pattern = F.conv2d(noise_pattern2 * mask, smoothfilter, padding=kernel_size // 2, groups=x.size(1))
            
            
            # Apply current noise pattern
            perturbed_image = x + smoothed_noise_pattern

            # Forward pass through the model
            output = model(perturbed_image)

            # Assuming 'output' is a dictionary with key 'x_hat' for the reconstructed image
            perturbed_output = output['x_hat']

            if qualitymeasure == 'psnr':
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                perturbed_quality = 10 * torch.log10((MAX_I ** 2) / mse_loss)

                # Compute the difference in PSNR between perturbed and target
                quality_loss = (perturbed_quality - target_quality).abs()


            elif qualitymeasure == 'ssim':
                # maximize distortion = minimize 1-SSIM
                perturbed_quality = ssim(perturbed_output, x)

                quality_loss = (perturbed_quality - target_quality).abs() 


            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
                
            else:
                l1norm = 0

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # quality_loss_lambda = 0.1
            
            # Combine the losses
            # combined_loss = -bpploss + quality_loss_lambda * quality_loss + l1_lambda * l1norm
    
            combined_loss = -10*torch.log10(bpploss) + quality_loss_lambda * torch.log10(quality_loss) + l1_lambda * l1norm


            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                
                print(f'Iteration {iteration} | {qualitymeasure}: {perturbed_quality} - Lost {quality_loss} | BPP {bpploss} |  Loss {combined_loss}')
                
    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': noise_pattern.data, \
                         'smoothed_noise_pattern': smoothed_noise_pattern.data,\
                         'perturbed_image': perturbed_image.data,\
                         'perturbed_output': perturbed_output.data,\
                         'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern
    
    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern




###_________________________________________####
###----------------------------------####
###----------------------------------####
def maxbitrate_logexp(x, errbound=0.1, smoothfilter=None, qualitymeasure='psnr', target_quality=None,quality_loss_lambda=0.1,l1_lambda=0, num_iterations=1000, model=None, device=None, mask=None,initial_noise=None,learningrate=1,keep_low_perturbation = False,keep_high_outcomequality = True):
    
    # Attack the whole image with a noise pattern which 
    # - maximizes the loss of the compression performance : maximize bpp 
    # - preserves the PSNR of the decompressed image :   min |PSNR(f(x + n)) - PSNR(f(x))|
    # - Sparse and smooth perturbed noise
    '''
        min_n -bpp(theta|n) + l_1 |PSNR(f(x + n)) - PSNR(f(x))| + ll_sparse ||x+n||
        s.t  |n_{i,kj}|<= sigma
    '''
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # qualitymeasure: type of quality metric to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # quality_loss_lambda :  weight for quality loss regularizer 
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1
        
    if learningrate is None:
        learningrate = 1

    if target_quality is None:
        target_quality = 0

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(device)
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    kappa = 5
    
    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            noise_pattern.data.clamp_(-errbound, errbound)
            noise_pattern2 = noise_pattern
#             noise_pattern2 = errbound*torch.tanh(noise_pattern)
 
            # Smooth the noise pattern
            if smoothfilter is None:
                smoothed_noise_pattern = noise_pattern2 * mask
            else:
                kernel_size = smoothfilter.shape[-1]
                smoothed_noise_pattern = F.conv2d(noise_pattern2 * mask, smoothfilter, padding=kernel_size // 2, groups=x.size(1))
            
            
            # Apply current noise pattern
            perturbed_image = torch.log(torch.exp(x) + smoothed_noise_pattern)

            # Forward pass through the model
            output = model(perturbed_image)

            # Assuming 'output' is a dictionary with key 'x_hat' for the reconstructed image
            perturbed_output = output['x_hat']

            if qualitymeasure == 'psnr':
                
                if keep_high_outcomequality == True:
                    # Calculate MSE loss(perturbed_output, x) : keep the decompressed image close to the original image
                    mse_loss = F.mse_loss(perturbed_output, x)

                    # Calculate PSNR loss
                    perturbed_quality = 10 * torch.log10((MAX_I ** 2) / mse_loss)

                    # Compute the difference in PSNR between perturbed and target
                    # quality_loss = torch.max((perturbed_quality - target_quality).abs()
                    quality_loss = torch.max((perturbed_quality - target_quality).abs()-2,torch.tensor(0.0))
                else:
                    quality_loss = 0
                    

                if keep_low_perturbation == True:
                    # Calculate MSE loss(perturbed_output, x) : keep the decompressed image close to the original image
                    mse_loss_ai = F.mse_loss(perturbed_image, x)

                    # Calculate PSNR loss
                    perturbed_quality_ai = 10 * torch.log10((MAX_I ** 2) / mse_loss_ai)

                    # Compute the difference in PSNR between perturbed and target
                    # quality_loss = torch.max((perturbed_quality - target_quality).abs()
                    quality_loss_ai = torch.max((perturbed_quality_ai - target_quality).abs()-2,torch.tensor(0.0))
                else:
                    quality_loss_ai=0
                
                # print(f'PSNR {perturbed_quality} {target_quality} {quality_loss}')
                
            elif qualitymeasure == 'ssim':
                # maximize distortion = minimize 1-SSIM
                perturbed_quality = ssim(perturbed_output, x)

                quality_loss = (perturbed_quality - target_quality).abs() 


            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
#                 print(l1norm)
            else:
                l1norm = 0

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # quality_loss_lambda = 0.1
            
            # Combine the losses
            # combined_loss = -bpploss + quality_loss_lambda * quality_loss + l1_lambda * l1norm
    
            combined_loss = -10*torch.log10(bpploss) + quality_loss_lambda * (quality_loss+quality_loss_ai) + l1_lambda * torch.log10(l1norm)


            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                
                print(f'Iteration {iteration} | {qualitymeasure}: {perturbed_quality} - Lost {quality_loss: .4f} | BPP {bpploss : .4f} |  Loss {combined_loss : .4f}')
                
    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': noise_pattern.data, \
                         'smoothed_noise_pattern': smoothed_noise_pattern.data,\
                         'perturbed_image': perturbed_image.data,\
                         'perturbed_output': perturbed_output.data,\
                         'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern
    
    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern




def maxdistortion_logexpnoise(x, errbound=0.1, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=1000, \
                                      model=None, device=None, mask=None,initial_noise=None,learningrate=0.1):
    '''
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,     
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    ''' 
    #    
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    if learningrate is None:
        learningrate = 0.1
        
    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(device)
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()
    
            # Clamp the noise pattern values to ensure they stay within a valid range
            noise_pattern.data.clamp_(-errbound, errbound)
            # noise_pattern2 = errbound*torch.tanh(noise_pattern)
    
            # Smooth the noise pattern
            if smoothfilter is None:
                smoothed_noise_pattern = noise_pattern * mask
                #print(smoothed_noise_pattern)
            else:
                kernel_size = smoothfilter.shape[-1]
                smoothed_noise_pattern = F.conv2d(noise_pattern * mask, smoothfilter, padding=kernel_size // 2, groups=x.size(1))
                
            # Apply current noise pattern
            perturbed_image = torch.log(torch.exp(x) + smoothed_noise_pattern)
            #print(perturbed_image)
    
            # Forward pass through the model
            # output = model(perturbed_image)
            output = model.forward(perturbed_image)
    
            # output['x_hat'] for the reconstructed image
            perturbed_output = output['x_hat']
    
            if losstype == 'psnr':
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)
    
                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I ** 2) / mse_loss)
    
                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR
    
            elif losstype == 'ssim':
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)
    
                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM
    
            elif losstype == 'dists':
                
                dists_perturbed = dists(x, perturbed_output)    
                loss = -dists_perturbed    
    
            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm
            else:
                combined_loss = loss
    
            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)
    
            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()
    
            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                print(f'Iteration {iteration}, Loss: {loss.item(): .4f}, BPP {bpploss : .4f}')
    
 
    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': noise_pattern.data, \
                         'smoothed_noise_pattern': smoothed_noise_pattern.data,\
                         'perturbed_image': perturbed_image.data,\
                         'perturbed_output': perturbed_output.data,\
                         'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

## LOGEXP noise model 
###_________________________________####
 
def maxdistortion_logexpnoise_tanh(x, errbound=0.1, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=1000, model=None, device=None, mask=None,initial_noise=None,learningrate=0.1):
    '''
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,     
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    ''' 
    #    
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    if learningrate is None:
        learningrate = 0.1
        
    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(device)
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()
    
            # Clamp the noise pattern values to ensure they stay within a valid range
            #noise_pattern.data.clamp_(-errbound, errbound)
            # noise_pattern2 = errbound*torch.tanh(noise_pattern)
    
            #smoothed_noise_pattern = errbound * torch.tanh(noise_pattern) * mask
    
            # Smooth the noise pattern
            if smoothfilter is None:
                smoothed_noise_pattern = errbound * torch.tanh(noise_pattern) * mask
                #print(smoothed_noise_pattern)
            else:
                kernel_size = smoothfilter.shape[-1]
                smoothed_noise_pattern = F.conv2d(errbound * torch.tanh(noise_pattern) * mask, smoothfilter, padding=kernel_size // 2, groups=x.size(1))
                
            # Apply current noise pattern
            perturbed_image = torch.log(torch.exp(x) + smoothed_noise_pattern)
            #print(perturbed_image)
    
            # Forward pass through the model
            output = model(perturbed_image)
    
            # output['x_hat'] for the reconstructed image
            perturbed_output = output['x_hat']
    
            if losstype == 'psnr':
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)
    
                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I ** 2) / mse_loss)
    
                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR
    
            elif losstype == 'ssim':
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)
    
                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM
    
            elif losstype == 'dists':
                
                dists_perturbed = dists(x, perturbed_output)    
                loss = -dists_perturbed    
    
            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm
            else:
                combined_loss = loss
    
            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)
    
            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()
    
            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                print(f'Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}')
    
 
    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': noise_pattern.data, \
                         'smoothed_noise_pattern': smoothed_noise_pattern.data,\
                         'perturbed_image': perturbed_image.data,\
                         'perturbed_output': perturbed_output.data,\
                         'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern
 
    
def maxdistortion_logexpnoise(x, errbound=0.1, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=1000, model=None, device=None, mask=None,initial_noise=None,learningrate=0.1):
    '''
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,     
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    ''' 
    #    
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    if learningrate is None:
        learningrate = 0.1
        
    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(device)
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()
    
            # Clamp the noise pattern values to ensure they stay within a valid range
            noise_pattern.data.clamp_(-errbound, errbound)
            # noise_pattern2 = errbound*torch.tanh(noise_pattern)
    
            # Smooth the noise pattern
            if smoothfilter is None:
                smoothed_noise_pattern = noise_pattern * mask
                #print(smoothed_noise_pattern)
            else:
                kernel_size = smoothfilter.shape[-1]
                smoothed_noise_pattern = F.conv2d(noise_pattern * mask, smoothfilter, padding=kernel_size // 2, groups=x.size(1))
                
            # Apply current noise pattern
            perturbed_image = torch.log(torch.exp(x) + smoothed_noise_pattern)
            #print(perturbed_image)
    
            # Forward pass through the model
            output = model(perturbed_image)
    
            # output['x_hat'] for the reconstructed image
            perturbed_output = output['x_hat']
    
            if losstype == 'psnr':
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)
    
                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I ** 2) / mse_loss)
    
                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR
    
            elif losstype == 'ssim':
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)
    
                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM
    
            elif losstype == 'dists':
                
                dists_perturbed = dists(x, perturbed_output)    
                loss = -dists_perturbed    
    
            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm
            else:
                combined_loss = loss
    
            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)
    
            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()
    
            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                print(f'Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}')
    
 
    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': noise_pattern.data, \
                         'smoothed_noise_pattern': smoothed_noise_pattern.data,\
                         'perturbed_image': perturbed_image.data,\
                         'perturbed_output': perturbed_output.data,\
                         'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern
 
    
###_________________________________####
    
### For multiscale wavelets attack

import torch
import pytorch_wavelets
import torch.nn as nn

class MultiScaleDecomposition:
    def __init__(self, wavelet='haar', device='cpu', scales = 1):
        self.dwt = pytorch_wavelets.DWTForward(J=scales, wave=wavelet).to(device)
        self.device = device

    def decompose(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        # Decompose x into low and high-frequency components
        low, details = self.dwt(x)
        return low, details



class Reconstruction:
    def __init__(self, wavelet='haar', device='cpu'):
        self.idwt = pytorch_wavelets.DWTInverse(wave=wavelet).to(device)
        self.device = device

    def reconstruct(self, low, perturbed_details):
        # Ensure inputs are on the correct device
        low = low.to(self.device)
        perturbed_details = [d.to(self.device) for d in perturbed_details]
        # Reconstruct image
        return self.idwt((low, perturbed_details))

    

###_________________________________####
    


def maxdistortion_logexp_multiscale(x, errbound=0.1, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=1000, \
                                      model=None, device=None, mask=None,initial_noise=None,learningrate=0.1,scales = 1, wavelet = 'haar'):
    # Multiscale log-exp noise attack
    # 
    '''
    \min_{n}  PSNR(x_{out},x_{in}) + \lambda  ||n||_1,     
        s.t.   x_{out} = f(x_attack), |noise_{ij}|<= \sigma
        
    where x_attack = idwt(logexp(dwt(x)+noise))
    #
    ''' 
    #
    #    
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # Set default learning rate if not provided
    if learningrate is None:
        learningrate = 0.1
         
    # Initialize components
    decomposer = MultiScaleDecomposition(wavelet=wavelet, device=device, scales=scales)
    reconstructor = Reconstruction(wavelet=wavelet, device=device)

    # Decompose input image
    low, details = decomposer.decompose(x)
    
    # Если mask=None, то делаем единичные маски для каждого уровня
    if mask is None:
        mask = [torch.ones_like(low)] + [torch.ones_like(d) for d in details]
    else:
        mask = [m.to(device) for m in mask]  # убедимся, что все на GPU
    
    # Initialize the noise pattern
    if initial_noise is None:
        # Generate random tensors matching the size of each 'details[k]'
        low_noise_pattern = torch.nn.Parameter(torch.randn_like(low))
        detail_noise_pattern = [
            torch.nn.Parameter(torch.randn_like(d)) for d in details
        ]
    else:
        # Use provided initial noise scaled by mask
        detail_noise_pattern = [torch.nn.Parameter(initial_noise[k+1].detach().clone() * mask[k+1]) for k in range(scales)]
        low_noise_pattern = torch.nn.Parameter(initial_noise[0].detach().clone()) 
        
    # Combine all parameters into a single list
    noise_pattern = [low_noise_pattern] + detail_noise_pattern

    # Move noise_pattern tensors to the appropriate device
    noise_pattern = [param.to(device) for param in noise_pattern]

    # Optional: Print the shapes for verification
    for k, param in enumerate(noise_pattern):
        print(f"Scale {k} - Noise pattern shape: {param.shape}")

    # Set error bound
    delta = errbound

    # Define optimizer 
    optimizer = torch.optim.SGD(noise_pattern, lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
    
    try:
        # Initialize lists to store masked noise patterns and perturbed details
        smoothed_noise_pattern = [None] * (scales+1)  # List to hold masked noise patterns
        perturbed_details = [None] * (scales)      # List to hold perturbed details

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            for k in range(scales+1):
                # Clamp the noise pattern values to ensure they stay within a valid range
                noise_pattern[k].data.clamp_(-errbound, errbound)

                # Apply masks to the clamped noise patterns, ensuring masks do not track gradients
                smoothed_noise_pattern[k] = noise_pattern[k] * mask[k]

                # Apply current noise pattern to the details
                if k == 0:
                    perturbed_low = torch.sign(low) * torch.log(torch.exp(low.abs()) + smoothed_noise_pattern[k])
                else:
                    d_old = details[k-1]
                    perturbed_details[k-1] = (
                        torch.sign(d_old) *
                        torch.log(torch.exp(d_old.abs()) + smoothed_noise_pattern[k])
                    )
            
            # Reconstruct the perturbed image from the low-frequency component and perturbed details
            perturbed_image = reconstructor.reconstruct(perturbed_low, perturbed_details)

            # Forward pass through the model
            output = model(perturbed_image) 
    
            # output['x_hat'] for the reconstructed image
            perturbed_output = output['x_hat']
    
            if losstype == 'psnr':
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)
    
                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I ** 2) / mse_loss)
    
                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR
    
            elif losstype == 'ssim':
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)
    
                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM
    
            elif losstype == 'dists':
                
                dists_perturbed = dists(x, perturbed_output)    
                loss = -dists_perturbed    
    
            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm  
            else:
                combined_loss = loss  
    
            
            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)
    
            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()
    

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                # Optionally, increase alpha gradually
                print(f'Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}')
    
 
    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': [tensor.detach().cpu().numpy() for tensor in noise_pattern], \
                         'smoothed_noise_pattern': [tensor.detach().cpu().numpy() for tensor in smoothed_noise_pattern],\
                         'perturbed_image': perturbed_image.data,\
                         'perturbed_output': perturbed_output.data,\
                         'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern
 
    
def maxdistortion_logexp_multiscale_old(x, errbound=0.1, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=1000, \
                                      model=None, device=None, mask=None,initial_noise=None,learningrate=0.1,scales = 1,wavelet = 'haar'):
    # Multiscale log-exp noise attack
    # 
    '''
    \min_{n}  PSNR(x_{out},x_{in}) + \lambda  ||n||_1,     
        s.t.   x_{out} = f(x_attack), |noise_{ij}|<= \sigma
        
    where x_attack = idwt(logexp(dwt(x)+noise))
    #
    ''' 
    #
    #    
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = [torch.ones(1) for _ in range(scales+1)] 

    # Move mask to the appropriate device
    mask = [m.to(device).detach() for m in mask]


    # Set default learning rate if not provided
    if learningrate is None:
        learningrate = 0.1
         
        
    # Initialize components
    # wavelet = 'haar'
    decomposer = MultiScaleDecomposition(wavelet=wavelet, device=device,scales = scales)
    reconstructor = Reconstruction(wavelet=wavelet, device=device)

    # Decompose input image
    low, details = decomposer.decompose(x)
    
    # Initialize the noise pattern
    if initial_noise is None:
        # Generate random tensors matching the size of each 'details[k]'
        detail_noise_pattern = [torch.nn.Parameter(torch.randn_like(detail) * mask[k+1]) for k, detail in enumerate(details)]
        low_noise_pattern = torch.nn.Parameter(torch.randn_like(low)) 
        
        
    else:
        # Use provided initial noise scaled by mask
        detail_noise_pattern = [torch.nn.Parameter(initial_noise[k+1].detach().clone() * mask[k+1]) for k in range(scales)]
        low_noise_pattern = torch.nn.Parameter(initial_noise[0].detach().clone()) 
        
    # Combine all parameters into a single list
    noise_pattern = [low_noise_pattern] + detail_noise_pattern


    # Move noise_pattern tensors to the appropriate device
    noise_pattern = [param.to(device) for param in noise_pattern]


    # Optional: Print the shapes for verification
    for k, param in enumerate(noise_pattern):
        print(f"Scale {k} - Noise pattern shape: {param.shape}")

    # Set error bound
    delta = errbound


    # Define optimizer 
    optimizer = torch.optim.SGD(noise_pattern, lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
    
    
#     low[:] = 0
    
    try:
        
        # Initialize lists to store masked noise patterns and perturbed details
        smoothed_noise_pattern = [None] * (scales+1)  # List to hold masked noise patterns
        perturbed_details = [None] * (scales)      # List to hold perturbed details
#         perturbed_low = [None] *      # List to hold perturbed details

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            for k in range(scales+1):
                # Clamp the noise pattern values to ensure they stay within a valid range
                noise_pattern[k].data.clamp_(-errbound, errbound)

                # Apply masks to the clamped noise patterns, ensuring masks do not track gradients
                smoothed_noise_pattern[k] = noise_pattern[k] * mask[k]

                # Apply current noise pattern to the details
                if k == 0:
                    perturbed_low = torch.log(torch.exp(low) + smoothed_noise_pattern[k])
                else:
                    perturbed_details[k-1] = torch.log(torch.exp(details[k-1]) + smoothed_noise_pattern[k])
            
 
            # Reconstruct the perturbed image from the low-frequency component and perturbed details
            perturbed_image = reconstructor.reconstruct(perturbed_low, perturbed_details)

            
            # Forward pass through the model
            output = model(perturbed_image) 
    
            # output['x_hat'] for the reconstructed image
            perturbed_output = output['x_hat']
    
            if losstype == 'psnr':
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)
    
                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I ** 2) / mse_loss)
    
                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR
    
            elif losstype == 'ssim':
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)
    
                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM
    
            elif losstype == 'dists':
                
                dists_perturbed = dists(x, perturbed_output)    
                loss = -dists_perturbed    
    
            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm  
            else:
                combined_loss = loss  
    
            
            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)
    
            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()
    

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                # Optionally, increase alpha gradually
                print(f'Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}')
    
 
    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': [tensor.detach().cpu().numpy() for tensor in noise_pattern], \
                         'smoothed_noise_pattern': [tensor.detach().cpu().numpy() for tensor in smoothed_noise_pattern],\
                         'perturbed_image': perturbed_image.data,\
                         'perturbed_output': perturbed_output.data,\
                         'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern
 

def maxdistortion_logexp_tanh_multiscale(x, errbound=0.1, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=1000, \
                                      model=None, device=None, mask=None,initial_noise=None,learningrate=0.1,scales = 1,wavelet = 'haar'):
    # Multiscale log-exp noise attack
    # 
    '''
    \min_{n}  PSNR(x_{out},x_{in}) + \lambda  ||n||_1,     
        s.t.   x_{out} = f(x_attack), |noise_{ij}|<= \sigma
        
    where x_attack = idwt(logexp(dwt(x)+noise))
    #
    ''' 
    #
    #    
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = [torch.ones(1) for _ in range(scales+1)] 

    # Move mask to the appropriate device
    mask = [m.to(device).detach() for m in mask]


    # Set default learning rate if not provided
    if learningrate is None:
        learningrate = 0.1
         
        
    # Initialize components
    # wavelet = 'haar'
    decomposer = MultiScaleDecomposition(wavelet=wavelet, device=device,scales = scales)
    reconstructor = Reconstruction(wavelet=wavelet, device=device)

    # Decompose input image
    low, details = decomposer.decompose(x)
    
    # Initialize the noise pattern
    if initial_noise is None:
        # Generate random tensors matching the size of each 'details[k]'
        detail_noise_pattern = [torch.nn.Parameter(torch.randn_like(detail) * mask[k+1]) for k, detail in enumerate(details)]
        low_noise_pattern = torch.nn.Parameter(torch.randn_like(low)) 
        
        
    else:
        # Use provided initial noise scaled by mask
        detail_noise_pattern = [torch.nn.Parameter(initial_noise[k+1].detach().clone() * mask[k+1]) for k in range(scales)]
        low_noise_pattern = torch.nn.Parameter(initial_noise[0].detach().clone()) 
        
    # Combine all parameters into a single list
    noise_pattern = [low_noise_pattern] + detail_noise_pattern


    # Move noise_pattern tensors to the appropriate device
    noise_pattern = [param.to(device) for param in noise_pattern]


    # Optional: Print the shapes for verification
    for k, param in enumerate(noise_pattern):
        print(f"Scale {k} - Noise pattern shape: {param.shape}")

    # Set error bound
    delta = errbound


    # Define optimizer 
    optimizer = torch.optim.SGD(noise_pattern, lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
    
    
#     low[:] = 0
    
    try:
        
        # Initialize lists to store masked noise patterns and perturbed details
        smoothed_noise_pattern = [None] * (scales+1)  # List to hold masked noise patterns
        perturbed_details = [None] * (scales)      # List to hold perturbed details
#         perturbed_low = [None] *      # List to hold perturbed details

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            for k in range(scales+1):
                # Clamp the noise pattern values to ensure they stay within a valid range
                # noise_pattern[k].data.clamp_(-errbound, errbound)

                # Apply masks to the clamped noise patterns, ensuring masks do not track gradients
                #smoothed_noise_pattern[k] = noise_pattern[k] * mask[k]
                
                smoothed_noise_pattern[k] = errbound * torch.tanh(noise_pattern[k]) * mask[k]

                # Apply current noise pattern to the details
                if k == 0:
                    perturbed_low = torch.log(torch.exp(low) + smoothed_noise_pattern[k])
                else:
                    perturbed_details[k-1] = torch.log(torch.exp(details[k-1]) + smoothed_noise_pattern[k])
            
 
            # Reconstruct the perturbed image from the low-frequency component and perturbed details
            perturbed_image = reconstructor.reconstruct(perturbed_low, perturbed_details)

            
            # Forward pass through the model
            output = model(perturbed_image) 
    
            # output['x_hat'] for the reconstructed image
            perturbed_output = output['x_hat']
    
            if losstype == 'psnr':
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)
    
                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I ** 2) / mse_loss)
    
                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR
    
            elif losstype == 'ssim':
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)
    
                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM
    
            elif losstype == 'dists':
                
                dists_perturbed = dists(x, perturbed_output)    
                loss = -dists_perturbed    
    
            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm  
            else:
                combined_loss = loss  
    
            
            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)
    
            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()
    

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                # Optionally, increase alpha gradually
                print(f'Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}')
    
 
    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': [tensor.detach().cpu().numpy() for tensor in noise_pattern], \
                         'smoothed_noise_pattern': [tensor.detach().cpu().numpy() for tensor in smoothed_noise_pattern],\
                         'perturbed_image': perturbed_image.data,\
                         'perturbed_output': perturbed_output.data,\
                         'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern
 
    
###________________________________####
#  FOR DEFENSE 


##___________________________________________#######

def defend_maxdistortion_tanh(x, errbound, smoothfilter, losstype, l1_lambda, num_iterations, model, device=None, mask=None,initial_noise=None,learningrate = 0.1):
    '''
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,     
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    ''' 
    #    This depense works for other attack model
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1


    if learningrate is None:
        learningrate = 0.1
        
    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(device)
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()
    
            # Clamp the noise pattern values to ensure they stay within a valid range
            #noise_pattern.data.clamp_(-errbound, errbound)
            noise_pattern2 = errbound*torch.tanh(noise_pattern)
    
            # Smooth the noise pattern
            # kernel_size = smoothfilter.shape[-1]
            # smoothed_noise_pattern = F.conv2d(noise_pattern2 * mask, smoothfilter, padding=kernel_size // 2, groups=x.size(1))
            smoothed_noise_pattern = noise_pattern2 * mask
    
            # Apply current noise pattern
            perturbed_image = x + smoothed_noise_pattern
            
            # perturbed_image = torch.log(torch.exp(x) + smoothed_noise_pattern)
    
            # Forward pass through the model
            output = model(perturbed_image)
    
            # output['x_hat'] for the reconstructed image
            perturbed_output = output['x_hat']
    
            if losstype == 'psnr':
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, perturbed_image)
    
                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I ** 2) / mse_loss)
    
                # maximize distortion = minimize PSNR
                loss = -psnr_loss  # Negative sign because we want to maximize PSNR
    
            elif losstype == 'ssim':
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)
    
                loss = ssim_perturbed  # Negative sign because we want to maximize SSIM
    
            elif losstype == 'dists':
                
                dists_perturbed = dists(x, perturbed_output)    
                loss = dists_perturbed    
    
            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm
            else:
                combined_loss = loss
    
            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)
    
            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()
    
            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                print(f'Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}')

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': noise_pattern.data, \
                         'smoothed_noise_pattern': smoothed_noise_pattern.data,\
                         'perturbed_image': perturbed_image.data,\
                         'perturbed_output': perturbed_output.data,\
                         'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


##___________________________________________#######
def defend_maxbpp_tanh(x, errbound=0.1, smoothfilter=None, qualitymeasure='psnr',
                       target_quality=None,quality_loss_lambda=0.1,l1_lambda=0, 
                       num_iterations=1000, model=None, device=None, mask=None,initial_noise=None,learningrate=1):


    '''
        min_n bpp(theta|n) + l_1 |PSNR(f(x + n)) - PSNR(f(x))| + ll_sparse ||x+n||
        s.t  |n_{i,kj}|<= sigma
    '''
    # x: (infected) attacked input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # qualitymeasure: type of quality metric to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # quality_loss_lambda :  weight for quality loss regularizer 
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan 
   
    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1


    if learningrate is None:
        learningrate = 0.1
        
    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(device)
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()
    
            # Clamp the noise pattern values to ensure they stay within a valid range
            #noise_pattern.data.clamp_(-errbound, errbound)
            noise_pattern2 = errbound*torch.tanh(noise_pattern)
    
            # Smooth the noise pattern
            # kernel_size = smoothfilter.shape[-1]
            # smoothed_noise_pattern = F.conv2d(noise_pattern2 * mask, smoothfilter, padding=kernel_size // 2, groups=x.size(1))
            smoothed_noise_pattern = noise_pattern2 * mask
    
            # Apply current noise pattern
            perturbed_image = x + smoothed_noise_pattern
            
            # perturbed_image = torch.log(torch.exp(x) + smoothed_noise_pattern)
    
            # Forward pass through the model
            output = model(perturbed_image)
    
            # output['x_hat'] for the reconstructed image
            perturbed_output = output['x_hat']
    
            if qualitymeasure == 'psnr':
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                perturbed_quality = 10 * torch.log10((MAX_I ** 2) / mse_loss)

                # Compute the difference in PSNR between perturbed and target
                quality_loss = (perturbed_quality - target_quality).abs()


            elif qualitymeasure == 'ssim':
                # maximize distortion = minimize 1-SSIM
                perturbed_quality = ssim(perturbed_output, x)

                quality_loss = (perturbed_quality - target_quality).abs() 


            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
                
            else:
                l1norm = 0

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # quality_loss_lambda = 0.1
            
            # Combine the losses
            # combined_loss = -bpploss + quality_loss_lambda * quality_loss + l1_lambda * l1norm
    
            combined_loss = 10*torch.log10(bpploss) + quality_loss_lambda * torch.log10(quality_loss) + l1_lambda * l1norm
 
            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()
    
            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                
                print(f'Iteration {iteration} | {qualitymeasure}: {perturbed_quality} - Lost {quality_loss} | BPP {bpploss} |  Loss {combined_loss}')
                
    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': noise_pattern.data, \
                         'smoothed_noise_pattern': smoothed_noise_pattern.data,\
                         'perturbed_image': perturbed_image.data,\
                         'perturbed_output': perturbed_output.data,\
                         'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern
    
    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

from kornia.losses import SSIMLoss, PSNRLoss
psnr = PSNRLoss(max_val=1.0)

####_______________________________###
####_______________________________###
def exec_refine_noiselevel_logexp_tanh(x,perturbed_output, perturbed_image, errbound=0.1, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=1000,quality_loss_lambda= 0.1, \
                           model=None, device=None, mask=None,noise_pattern=None,learningrate=0.1,scales = 1,target_quality = 100,desired_noiselevel = 0.02,wavelet = 'haar'):

    # REduce the noise level from the intial noise level to a desired noise bound 
    # 
    # Anh-Huy Phan 
    # Initialize variables
    psnr_ = [psnr(perturbed_output, perturbed_image).detach().cpu().numpy()]
    attack_area_ = []

    # quality_loss_lambda = 0.1
      
    current_noiselevel = errbound
    sigmas = np.linspace(current_noiselevel, desired_noiselevel, 4)

    # Variables to keep track of the last positive psnr_k and its corresponding parameters
    last_positive_psnr_k = None
    last_positive_attack_area = None
    last_positive_params = None
    
    # number of inner runs to refine the noise 
    inner_runs = 5
    
    try:

        for current_noiselevel_ in sigmas:
            print(current_noiselevel_)

            for krun in range(inner_runs):
                noise_pattern_bk = [param.clone() for param in noise_pattern]

                perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
                    maxdistortion_logexp_multiscale(x, errbound=current_noiselevel_, smoothfilter = smoothfilter, \
                                                    losstype = losstype, l1_lambda=l1_lambda, num_iterations=num_iterations, \
                                                      model=model, device=device, mask=mask,initial_noise=noise_pattern,learningrate=learningrate,scales = scales,wavelet = wavelet)

                psnr_k = psnr(perturbed_output,perturbed_image).detach().cpu().numpy()


                print(f'{psnr_k}')
                psnr_.append(psnr_k)

                if psnr_k > 0:
                    last_positive_psnr_k = psnr_k
        #             last_positive_attack_area = new_attack_area
                    last_positive_params = {
                        'perturbed_image': perturbed_image.clone(),
                        'perturbed_output': perturbed_output.clone(),
                        'smoothed_noise_pattern': [param.clone() for param in smoothed_noise_pattern],
                        'noise_pattern':  [param.clone() for param in noise_pattern],
                        'mask': None,
                        'current_noiselevel': current_noiselevel_
                    }

                if psnr_k < 50:
                    noise_pattern = [param.clone() for param in noise_pattern_bk]

                    current_noiselevel_ = last_positive_params["current_noiselevel"]

                    perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
                    maxdistortion_logexp_tanh_multiscale(x, errbound=current_noiselevel_, smoothfilter = None, \
                                losstype = losstype, l1_lambda=l1_lambda, num_iterations=num_iterations, \
                                  model=model, device=device, mask=None,initial_noise=noise_pattern,learningrate=0.0001,scales = scales,wavelet = wavelet)

                    break

                if psnr_k < 0:
                    break

            if psnr_k < 0:
                break

        # Save the results to the original variables
        if last_positive_params:
            perturbed_image = last_positive_params['perturbed_image']
            perturbed_output = last_positive_params['perturbed_output']
            smoothed_noise_pattern = last_positive_params['smoothed_noise_pattern']
            noise_pattern = last_positive_params['noise_pattern']
            mask = last_positive_params['mask']

        
    except KeyboardInterrupt:
         print("Interrupted, stop and return the current results.")
         return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern,current_noiselevel_,psnr_
    
    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern, current_noiselevel_,psnr_


def exec_minpsnr_attack_logexp_tanh_multiscales(x, errbound=0.1, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=1000, \
                                      model=None, device=None, mask=None,initial_noise=None,learningrate=0.1,scales = 1,target_quality = 100,wavelet = 'haar'):


    '''
        min_n bpp(theta|n) + l_1 |PSNR(f(x + n)) - PSNR(f(x))| + ll_sparse ||x+n||
        s.t  |n_{i,kj}|<= sigma
    '''
    # x: (infected) attacked input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # qualitymeasure: type of quality metric to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # quality_loss_lambda :  weight for quality loss regularizer 
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan 
   
    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device
         

    if learningrate is None:
        learningrate = 0.001
        
    noise_pattern = initial_noise
    
    try:
        psnr_k = 0  # Initialize PSNR to enter the loop


        while psnr_k < target_quality: 

            perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
                maxdistortion_logexp_tanh_multiscale(x, errbound=errbound, smoothfilter = smoothfilter, losstype = losstype, \
                l1_lambda=l1_lambda, num_iterations=num_iterations, model=model, device=device,\
                mask=mask,initial_noise=noise_pattern,learningrate=learningrate,scales = scales,wavelet = wavelet)



            
#             perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
#                 maxdistortion_logexp_multiscale(x, errbound=current_noiselevel, smoothfilter = smoothfilter,\
#                 losstype = qualitymeasure, l1_lambda=l1_lambda, num_iterations=num_iterations, model=model, \
#                 device=device, mask=mask,initial_noise=noise_pattern,learningrate=learningrate,scales = scales)

            psnr_k = psnr(perturbed_output,perturbed_image).detach().cpu().numpy()

     
    except KeyboardInterrupt:
         print("Interrupted, stop and return the current results.")
         return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern
    
    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern



def exec_minpsnr_attack_logexp_multiscales(x, errbound=0.1, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=1000, \
                                      model=None, device=None, mask=None,initial_noise=None,learningrate=0.1,scales = 1,target_quality = 100,wavelet = 'haar'):


    '''
        min_n bpp(theta|n) + l_1 |PSNR(f(x + n)) - PSNR(f(x))| + ll_sparse ||x+n||
        s.t  |n_{i,kj}|<= sigma
    '''
    # x: (infected) attacked input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # qualitymeasure: type of quality metric to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # quality_loss_lambda :  weight for quality loss regularizer 
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan 
   
    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device
         

    if learningrate is None:
        learningrate = 0.001
        
    noise_pattern = initial_noise
    
    try:
        psnr_k = 0  # Initialize PSNR to enter the loop


        while psnr_k < target_quality: 

            perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
                maxdistortion_logexp_multiscale(x, errbound=errbound, smoothfilter = smoothfilter, losstype = losstype, \
                l1_lambda=l1_lambda, num_iterations=num_iterations, model=model, device=device,\
                mask=mask,initial_noise=noise_pattern,learningrate=learningrate,scales = scales,wavelet = wavelet)



            
#             perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
#                 maxdistortion_logexp_multiscale(x, errbound=current_noiselevel, smoothfilter = smoothfilter,\
#                 losstype = qualitymeasure, l1_lambda=l1_lambda, num_iterations=num_iterations, model=model, \
#                 device=device, mask=mask,initial_noise=noise_pattern,learningrate=learningrate,scales = scales)

            psnr_k = psnr(perturbed_output,perturbed_image).detach().cpu().numpy()

     
    except KeyboardInterrupt:
         print("Interrupted, stop and return the current results.")
         return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern
    
    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


def exec_refine_noiselevel(x,perturbed_output, perturbed_image, errbound=0.1, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=1000,quality_loss_lambda= 0.1, \
                           model=None, device=None, mask=None,noise_pattern=None,learningrate=0.1,scales = 1,target_quality = 100,desired_noiselevel = 0.02,wavelet = 'haar'):

    # REduce the noise level from the intial noise level to a desired noise bound 
    # 
    # Anh-Huy Phan 
    # Initialize variables
    psnr_ = [psnr(perturbed_output, perturbed_image).detach().cpu().numpy()]
    attack_area_ = []

    # quality_loss_lambda = 0.1
      
    current_noiselevel = errbound
    sigmas = np.linspace(current_noiselevel, desired_noiselevel, 4)

    # Variables to keep track of the last positive psnr_k and its corresponding parameters
    last_positive_psnr_k = None
    last_positive_attack_area = None
    last_positive_params = None
    
    # number of inner runs to refine the noise 
    inner_runs = 5
    
    try:

        for current_noiselevel_ in sigmas:
            print(current_noiselevel_)

            for krun in range(inner_runs):
                noise_pattern_bk = [param.clone() for param in noise_pattern]

                perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
                    maxdistortion_logexp_multiscale(x, errbound=current_noiselevel_, smoothfilter = smoothfilter, \
                                                    losstype = losstype, l1_lambda=l1_lambda, num_iterations=num_iterations, \
                                                      model=model, device=device, mask=mask,initial_noise=noise_pattern,learningrate=learningrate,scales = scales,wavelet = wavelet)

                psnr_k = psnr(perturbed_output,perturbed_image).detach().cpu().numpy()


                print(f'{psnr_k}')
                psnr_.append(psnr_k)

                if psnr_k > 0:
                    last_positive_psnr_k = psnr_k
        #             last_positive_attack_area = new_attack_area
                    last_positive_params = {
                        'perturbed_image': perturbed_image.clone(),
                        'perturbed_output': perturbed_output.clone(),
                        'smoothed_noise_pattern': [param.clone() for param in smoothed_noise_pattern],
                        'noise_pattern':  [param.clone() for param in noise_pattern],
                        'mask': None,
                        'current_noiselevel': current_noiselevel_
                    }

                if psnr_k < 50:
                    noise_pattern = [param.clone() for param in noise_pattern_bk]

                    current_noiselevel_ = last_positive_params["current_noiselevel"]

                    perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
                    maxdistortion_logexp_multiscale(x, errbound=current_noiselevel_, smoothfilter = smoothfilter, \
                                losstype = losstype, l1_lambda=l1_lambda, num_iterations=num_iterations, \
                                  model=model, device=device, mask=None,initial_noise=noise_pattern,learningrate=learningrate,scales = scales,wavelet = wavelet)

                    break

                if psnr_k < 0:
                    break

            if psnr_k < 0:
                break

        # Save the results to the original variables
        if last_positive_params:
            perturbed_image = last_positive_params['perturbed_image']
            perturbed_output = last_positive_params['perturbed_output']
            smoothed_noise_pattern = last_positive_params['smoothed_noise_pattern']
            noise_pattern = last_positive_params['noise_pattern']
            mask = last_positive_params['mask']

        
    except KeyboardInterrupt:
         print("Interrupted, stop and return the current results.")
         return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern,current_noiselevel_,psnr_
    
    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern, current_noiselevel_,psnr_

import torch.nn.functional as F

def allocate_attack_mask(perturbed_image,perturbed_output):
    # Allocation the effective noise region 
    # Anh-Huy Phan
    #
    device = perturbed_image.device
    # Gaussian blur to the mask to smooth the edges
    kernel_size = 21
    sigma_filter = 8

    # Create the Gaussian kernel
    gaussian_filter_mask = gaussian_kernel(kernel_size, sigma_filter)

    # Add batch and channel dimensions to the filter
    gaussian_filter_mask = gaussian_filter_mask.view(1, 1, *gaussian_filter_mask.size())

    # Assuming 'image' is with shape [batch_size, channels, height, width]
    # Repeat the filter for each input channel
    gaussian_filter_mask = gaussian_filter_mask.repeat(perturbed_image.size(1), 1, 1, 1)
    gaussian_filter_mask = gaussian_filter_mask.to(device)


    # 
    residue = perturbed_output-perturbed_image
    # residue[torch.abs(residue)>0.5] = 1
    # residue =
    residue = F.conv2d(residue, gaussian_filter_mask, padding=gaussian_filter_mask.shape[2]//2,groups=perturbed_image.shape[1])
    residue = F.conv2d(residue, gaussian_filter_mask, padding=gaussian_filter_mask.shape[2]//2,groups=perturbed_image.shape[1])

    ell2_noise = torch.sqrt(torch.sum(residue**2,dim = 1));

    mask_noise = ell2_noise>.5#torch.max(ell2_noise) * 1e-1
    mask_noise = mask_noise.squeeze()

    mask_noise = mask_noise.cpu().detach().numpy()
    mask_noise = opening(mask_noise, square(10))
    # mask_noise = opening(mask_noise, square(10))
    mask_noise = dilation(mask_noise, square(20))
    # mask_noise = dilation(mask_noise, square(5))

    # plt.imshow(mask_noise)

    # Mask 3D 
    new_mask = torch.zeros_like(perturbed_image)
    nnz_ix = np.where(mask_noise==1)
    new_mask[:,:,nnz_ix[0],nnz_ix[1]] = 1

    # new_mask,new_2dmask = noise_to_mask(residue)
    new_mask = F.conv2d(new_mask, gaussian_filter_mask, padding=gaussian_filter_mask.shape[2]//2,groups=perturbed_image.shape[1])

    return new_mask

