"""
    Class that handles loading/saving of RAW/other formats.
    If images are of a regular filetype (jpg, png, ...); they are opened using opencv.
    Else use rawpy to load RAW image.
"""
import os
from io import BytesIO
import rawpy
import cv2
import imageio

# All RAW formats; src: https://fileinfo.com/filetypes/camera_raw
supported_rawpy_formats = [
    "RWZ",
    "RW2",
    "CR2",
    "DNG",
    "ERF",
    "NRW",
    "RAF",
    "ARW",
    "NEF",
    "K25",
    "DNG",
    "SRF",
    "EIP",
    "DCR",
    "RAW",
    "CRW",
    "3FR",
    "BAY",
    "MEF",
    "CS1",
    "KDC",
    "ORF",
    "ARI",
    "SR2",
    "MOS",
    "MFW",
    "CR3",
    "FFF",
    "SRW",
    "J6I",
    "X3F",
    "KC2",
    "RWL",
    "MRW",
    "PEF",
    "IIQ",
    "CXI",
    "MDC",
]
# Open-cv imread supported formats; src: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
supported_opencv_formats = [
    "bmp",
    "dib",
    "jpeg",
    "jpg",
    "jpe",
    "jp2",
    "png",
    "webp",
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",
    "pfm",
    "sr",
    "ras",
    "tiff",
    "tif",
    "exr",
    "hdr",
    "pic",
]


class ImageLoadingHandler:
    def __init__(self):
        return

    # Load src image to BGR 2D numpy array
    def read_image_from_path(self, path):
        # Get extension without dot at beginning
        _, extension = os.path.splitext(path)
        extension = extension[1:]
        if str.lower(extension) in supported_opencv_formats:
            # Regular imread
            return cv2.imread(path)
        elif str.upper(extension) in supported_rawpy_formats:
            # Read RAW image
            raw = rawpy.imread(path)

            processed = None
            try:
                # Extract thumbnail or preview (faster)
                thumb = raw.extract_thumb()
            except:
                # If no thumb/preview, then postprocess RAW image (slower)
                processed = raw.postprocess(use_camera_wb=True)
            else:
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    # Convert bytes object to ndarray
                    processed = imageio.imread(BytesIO(thumb.data))
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    # Ndarray
                    processed = thumb.data

            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

            raw.close()
            return processed
    
    # Get RAW image view from path (uses copy() to allow usage after closing raw file)
    def get_raw_view(self, path):
        raw = rawpy.imread(path)
        image = raw.raw_image_visible.copy()
        raw.close()
        return image
