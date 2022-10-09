"""
    Main pyramid stacking + image alignment algorithm(s).
"""
import os, tempfile, time
import cv2
import numpy as np
import numba as nb
from numba.typed import List

import src.algorithms.image_storage as image_storage
import src.algorithms.dft_imreg as dft_imreg
import src.algorithms.pyramid as pyramid_algorithm
import src.ImageLoadingHandler as ImageLoadingHandler
import src.settings as settings


# Pad an array to be the kernel size (square). Only if needed
@nb.njit(
    nb.float32[:, :](nb.float32[:, :], nb.int64),
    fastmath=True,
    cache=True,
)
def pad_array(array, kernel_size):
    y_shape = array.shape[0]
    x_shape = array.shape[1]

    y_pad = kernel_size - y_shape
    x_pad = kernel_size - x_shape
    if y_pad > 0 or x_pad > 0:
        # Pad array (copy values into new; larger array)
        padded_array = np.zeros((y_shape + y_pad, x_shape + x_pad), dtype=array.dtype)
        padded_array[0:y_shape, 0:x_shape] = array
        return padded_array
    else:
        # Don't do anything
        return array


# Get deviation of a (grayscale image) matrix
@nb.njit(
    nb.float32(nb.float32[:, :]),
    fastmath=True,
    cache=True,
)
def get_std_deviation(matrix):
    summed_deviation = float(0)
    average_value = np.mean(matrix)
    kernel_area = matrix.shape[0] * matrix.shape[1]

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            summed_deviation += (matrix[y, x] - average_value) ** 2 / kernel_area
    return np.sqrt(summed_deviation)


# Compute focusmap for the same pyramid level in 2 different pyramids
@nb.njit(
    nb.uint8[:, :](nb.float32[:, :], nb.float32[:, :], nb.int64),
    fastmath=True,
    parallel=True,
    cache=True,
)
def compute_focusmap(pyr_level1, pyr_level2, kernel_size):
    y_range = pyr_level1.shape[0]
    x_range = pyr_level1.shape[1]

    # 2D focusmap (dtype=uint8); possible values:
    # 0 => pixel of pyr1
    # 1 => pixel of pyr2
    focusmap = np.empty((y_range, x_range), dtype=np.uint8)
    k = int(kernel_size / 2)

    # Loop through pixels of this pyramid level
    for y in nb.prange(y_range):  # Most images are wider (more values on x-axis)
        for x in nb.prange(x_range):
            # Get small patch (kernel_size) around this pixel
            patch = pyr_level1[y - k : y + k, x - k : x + k]
            # Padd array with zeros if needed (edges of image)
            padded_patch = pad_array(patch, kernel_size)
            dev1 = get_std_deviation(padded_patch)

            patch = pyr_level2[y - k : y + k, x - k : x + k]
            padded_patch = pad_array(patch, kernel_size)
            dev2 = get_std_deviation(padded_patch)

            # Get entropy of kernel
            # deviation = entropy(padded_patch, disk(10))
            # print(kernel_entropy)

            value_to_insert = 0
            if dev2 > dev1:
                value_to_insert = 1

            # Write most in-focus pixel to output
            focusmap[y, x] = value_to_insert

    return focusmap


# Compute output pyramid_level from source arrays and focusmap
@nb.njit(
    nb.float32[:, :, :](nb.float32[:, :, :], nb.float32[:, :, :], nb.uint8[:, :]),
    fastmath=True,
    parallel=True,
    cache=True,
)
def fuse_pyramid_levels_using_focusmap(pyr_level1, pyr_level2, focusmap):
    # Copy directly in "pyr_level_1",
    # as creating a new array using ".copy()" takes longer
    for y in nb.prange(focusmap.shape[0]):
        for x in nb.prange(focusmap.shape[1]):
            if focusmap[y, x] == 0:
                pyr_level1[y, x, :] = pyr_level1[y, x, :]
            else:
                pyr_level1[y, x, :] = pyr_level2[y, x, :]
    return pyr_level1


class Algorithm:
    def __init__(self):
        self.ImageStorage = image_storage.ImageStorageHandler()
        self.ImageLoadingHandler = ImageLoadingHandler.ImageLoadingHandler()
        self.DFT_Imreg = dft_imreg.im_reg()

    def align_image_pair(self, ref_im, im_to_align):
        """
        Fast Fourier Transform (FFT) image translational registration ((x, y)-shift only!)
        'ref_im' and 'im_to_align' can be an image array (np.ndarray), or an image path (str).
        In the latter case, the images will be loaded into memory first.

        When both images are of type 'str', and they are the same,
        'im_to_align' will be loaded into memory and be returned without alignment.
        """
        if type(ref_im) == str and type(im_to_align) == str:
            return self.ImageLoadingHandler.read_image_from_path(im_to_align)

        if type(ref_im) == str:
            ref_im = self.ImageLoadingHandler.read_image_from_path(ref_im)
        if type(im_to_align) == str:
            im_to_align = self.ImageLoadingHandler.read_image_from_path(im_to_align)

        # Calculate translational shift
        # TODO: Allow adjusting "scale_factor"??
        return self.DFT_Imreg.register_image_translation(
            ref_im, im_to_align, scale_factor=10
        )

    def generate_laplacian_pyramid(self, im1, num_levels):
        """
        Generates a laplacian pyramid for each image.
        'im1' can be an image array (np.ndarray), or an image path (str).
        In the latter case, the image will be loaded into memory first.
        """
        if type(im1) == str:
            im1 = self.ImageLoadingHandler.read_image_from_path(im1)

        pyr1 = pyramid_algorithm.laplacian_pyramid(im1, num_levels)
        return pyr1

    # Fuse all sub-images of an image's Laplacian pyramid
    def focus_fuse_pyramids(self, image_archive_names, kernel_size, signals):
        output_pyramid = List()
        for i, archive_name in enumerate(image_archive_names):
            start_time = time.time()

            if i == 0:
                # Directly "copy" first image's pyramid into output
                laplacian_pyramid = self.ImageStorage.load_laplacian_pyramid(
                    archive_name
                )
                output_pyramid = laplacian_pyramid
            else:
                # Focus fuse this pyramid to the output
                new_laplacian_pyramid = self.ImageStorage.load_laplacian_pyramid(
                    archive_name
                )

                # Upscale last/largest focusmap (faster than computation)
                threshold_index = len(new_laplacian_pyramid) - 1
                new_pyr = List()
                current_focusmap = None
                # Loop through pyramid levels from smallest to largest shape
                for pyramid_level in range(len(new_laplacian_pyramid)):
                    if pyramid_level < threshold_index:
                        # Regular computation (slow; accurate)
                        current_focusmap = compute_focusmap(
                            cv2.cvtColor(
                                output_pyramid[pyramid_level], cv2.COLOR_BGR2GRAY
                            ),
                            cv2.cvtColor(
                                new_laplacian_pyramid[pyramid_level], cv2.COLOR_BGR2GRAY
                            ),
                            kernel_size,
                        )
                    else:
                        # TODO: See if upscale really provides any benefit
                        # Upscale previous mask (faster; less accurate)
                        s = new_laplacian_pyramid[pyramid_level].shape
                        current_focusmap = cv2.resize(
                            current_focusmap, (s[1], s[0]), interpolation=cv2.INTER_AREA
                        )

                    # Write using focusmap
                    new_pyr_level = fuse_pyramid_levels_using_focusmap(
                        output_pyramid[pyramid_level],
                        new_laplacian_pyramid[pyramid_level],
                        current_focusmap,
                    )

                    new_pyr.append(new_pyr_level)
                # Set updated pyramid
                output_pyramid = new_pyr

            # Send progress signals
            signals.finished_inter_task.emit(
                [
                    "laplacian_pyramid_focus_fusion",
                    i + 1,
                    len(image_archive_names),
                    time.time() - start_time,
                ]
            )

        return output_pyramid

    def focus_fuse_pyramid_pair(self, pyr1, pyr2, kernel_size):
        """
        Fuse 2 image pyramids into one.
        Each pyramid level will be compared between the two pyramids,
        and the sharpest pixels/parts of each image will be placed in the output pyramid.
        """
        # Upscale last/largest focusmap (faster than computation)
        threshold_index = len(pyr2) - 1
        # TODO: Check if Numba's 'List()' is needed? Use regular Python list instead?
        new_pyr = List()
        current_focusmap = None
        # Loop through pyramid levels from smallest to largest shape, and fuse each level
        for pyramid_level in range(len(pyr2)):
            # Calculate what parts are more/less in focus between the pyramids
            if pyramid_level < threshold_index:
                # Regular computation (slow; accurate)
                current_focusmap = compute_focusmap(
                    cv2.cvtColor(pyr1[pyramid_level], cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(pyr2[pyramid_level], cv2.COLOR_BGR2GRAY),
                    kernel_size,
                )
            else:
                # TODO: See if upscale really provides any benefit
                # Upscale previous mask (faster; less accurate)
                s = pyr2[pyramid_level].shape
                current_focusmap = cv2.resize(
                    current_focusmap, (s[1], s[0]), interpolation=cv2.INTER_AREA
                )

            # Write output pyramid level using the calculated focusmap
            new_pyr_level = fuse_pyramid_levels_using_focusmap(
                pyr1[pyramid_level],
                pyr2[pyramid_level],
                current_focusmap,
            )
            new_pyr.append(new_pyr_level)
        return new_pyr