"""
    Exposed API for easily aligning/stacking multiple images.
"""
import time
import numpy as np
import utilities as utilities
import algorithms as algorithms
import settings as settings

class LaplacianPyramid:
    def __init__(self, fusion_kernel_size=6, pyramid_num_levels=8):
        self.output_image = None
        self.image_paths = []

        self.Algorithm = algorithms.Algorithm()

        # Parameters
        self.fusion_kernel_size = fusion_kernel_size
        self.pyramid_num_levels = pyramid_num_levels

    def update_image_paths(self, new_image_paths):
        """
        Set new image paths (sorted by name).
        """
        self.image_paths = sorted(new_image_paths, key=utilities.int_string_sorting)


    def align_and_stack_images(self):
        """
        对齐和堆叠图像
        移除了 signals 参数和进度信号
        """
        aligned_images = [
            self.Algorithm.align_image_pair(self.image_paths[0], self.image_paths[0])
        ]
        fused_pyr = self.Algorithm.generate_laplacian_pyramid(
            aligned_images[0], self.pyramid_num_levels
        )

        for i, path in enumerate(self.image_paths):
            if i == 0:
                continue

            print(f"Processing image {i+1}/{len(self.image_paths)}...")
            
            aligned_images.append(
                self.Algorithm.align_image_pair(aligned_images[0], path)
            )
            new_pyr = self.Algorithm.generate_laplacian_pyramid(
                aligned_images[1], self.pyramid_num_levels
            )
            del aligned_images[0]
            
            fused_pyr = self.Algorithm.focus_fuse_pyramid_pair(
                fused_pyr, new_pyr, self.fusion_kernel_size
            )

        fused_image = self.Algorithm.reconstruct_pyramid(fused_pyr)
        self.output_image = fused_image
        return self.output_image

    def stack_images(self, signals):
        """
        Stack images.
        """
        # Will just load first image from path
        im0 = self.Algorithm.align_image_pair(self.image_paths[0], self.image_paths[0])
        fused_pyr = self.Algorithm.generate_laplacian_pyramid(
            im0,
            self.pyramid_num_levels,
        )

        for i, path in enumerate(self.image_paths):
            if i == 0:
                continue  # First image is already copied in pyr

            start_time = time.time()

            # Load from path
            im1 = self.Algorithm.align_image_pair(path, path)
            # Generate pyramid for the image
            new_pyr = self.Algorithm.generate_laplacian_pyramid(
                im1, self.pyramid_num_levels
            )
            # Delete image (lower memory usage)
            del im1
            # Fuse this new pyramid with the existing one
            fused_pyr = self.Algorithm.focus_fuse_pyramid_pair(
                fused_pyr, new_pyr, self.fusion_kernel_size
            )

            # Send progress signal
            signals.finished_inter_task.emit(
                [
                    "finished_image",
                    i + 1,
                    len(self.image_paths),
                    time.time() - start_time,
                ]
            )

        # Reconstruct image from Laplacian pyramid
        fused_image = self.Algorithm.reconstruct_pyramid(fused_pyr)
        self.output_image = fused_image
