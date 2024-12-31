import math
import cv2
import numpy as np
# import scipy.ndimage as ndi

def minimum_filter_numpy(array, size):
    """
    Apply a minimum filter to a 2D array using a sliding window approach.
    
    Parameters:
    - array: 2D numpy array to be filtered.
    - size: Size of the window (must be an odd integer).
    
    Returns:
    - filtered_array: 2D numpy array after applying the minimum filter.
    """
    # Ensure the size is odd
    if size % 2 == 0:
        raise ValueError("Size must be an odd integer.")
    
    # Pad the array to handle borders
    pad_width = size // 2
    padded_array = np.pad(array, pad_width, mode='edge')
    
    # Prepare an output array
    filtered_array = np.empty_like(array)
    
    # Iterate over each element in the array
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # Extract the local window
            local_window = padded_array[i:i+size, j:j+size]
            # Compute the minimum value in the local window
            filtered_array[i, j] = np.min(local_window)
    
    return filtered_array

def get_apofield(shape, aporad):
    """
    Returns an array between 0 and 1 that goes to zero close to the edges.
    """
    if aporad == 0:
        return np.ones(shape, dtype=float)
    apos = np.hanning(aporad * 2)
    vecs = []
    for dim in shape:
        assert dim > aporad * 2, "Apodization radius %d too big for shape dim. %d" % (
            aporad,
            dim,
        )
        toapp = np.ones(dim)
        toapp[:aporad] = apos[:aporad]
        toapp[-aporad:] = apos[-aporad:]
        vecs.append(toapp)
    apofield = np.outer(vecs[0], vecs[1])
    return apofield


def _argmax2D(array):
    """
    Simple 2D argmax function with simple sharpness indication
    """
    amax = np.argmax(array)
    ret = list(np.unravel_index(amax, array.shape))

    return np.array(ret)


def _phase_correlation(im0, im1, callback=None, *args):
    """
    Computes phase correlation between im0 and im1

    Args:
        im0
        im1
        callback (function): Process the cross-power spectrum (i.e. choose
            coordinates of the best element, usually of the highest one).
            Defaults to :func:`imreg_dft.utils.argmax2D`

    Returns:
        tuple: The translation vector (Y, X). Translation vector of (0, 0)
            means that the two images match.
    """
    if callback is None:
        callback = _argmax2D

    # # TODO: Implement some form of high-pass filtering of PHASE correlation
    # f0, f1 = [fft.fft2(arr) for arr in (im0, im1)]
    # # spectrum can be filtered (already),
    # # so we have to take precaution against dividing by 0
    # eps = abs(f1).max() * 1e-15
    # # cps == cross-power spectrum of im0 and im1
    # cps = abs(fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
    # # scps = shifted cps
    # scps = fft.fftshift(cps)

    # 计算二维傅里叶变换
    f0 = np.fft.fft2(im0)
    f1 = np.fft.fft2(im1)

    # 避免除以零
    eps = np.abs(f1).max() * 1e-15

    # 计算交叉功率谱，保持大小不变
    cps = np.fft.ifft2((f0 * f1.conjugate()) / (np.abs(f0) * np.abs(f1) + eps))

    # 取绝对值并保持大小不变
    cps = np.abs(cps)
    
    # 将交叉功率谱中心化
    scps = np.fft.fftshift(cps)

    (t0, t1), success = callback(scps, *args)
    ret = np.array((t0, t1))

    # _compensate_fftshift is not appropriate here, this is OK.
    t0 -= f0.shape[0] // 2
    t1 -= f0.shape[1] // 2

    ret -= np.array(f0.shape, int) // 2
    return ret, success


def _argmax_ext(array, exponent):
    """
    Calculate coordinates of the COM (center of mass) of the provided array.

    Args:
        array (ndarray): The array to be examined.
        exponent (float or 'inf'): The exponent we power the array with. If the
            value 'inf' is given, the coordinage of the array maximum is taken.

    Returns:
        np.ndarray: The COM coordinate tuple, float values are allowed!
    """

    # When using an integer exponent for _argmax_ext, it is good to have the
    # neutral rotation/scale in the center rather near the edges

    ret = None
    if exponent == "inf":
        ret = _argmax2D(array)
    else:
        col = np.arange(array.shape[0])[:, np.newaxis]
        row = np.arange(array.shape[1])[np.newaxis, :]

        arr2 = array**exponent
        arrsum = arr2.sum()
        if arrsum == 0:
            # We have to return SOMETHING, so let's go for (0, 0)
            return np.zeros(2)
        arrprody = np.sum(arr2 * col) / arrsum
        arrprodx = np.sum(arr2 * row) / arrsum
        ret = [arrprody, arrprodx]
        # We don't use it, but it still tells us about value distribution

    return np.array(ret)


def _get_subarr(array, center, rad):
    """
    Args:
        array (ndarray): The array to search
        center (2-tuple): The point in the array to search around
        rad (int): Search radius, no radius (i.e. get the single point)
            implies rad == 0
    """
    dim = 1 + 2 * rad
    subarr = np.zeros((dim,) * 2)
    corner = np.array(center) - rad
    for ii in range(dim):
        yidx = corner[0] + ii
        yidx %= array.shape[0]
        for jj in range(dim):
            xidx = corner[1] + jj
            xidx %= array.shape[1]
            subarr[ii, jj] = array[yidx, xidx]
    return subarr


def _interpolate(array, rough, rad=2):
    """
    Returns index that is in the array after being rounded.

    The result index tuple is in each of its components between zero and the
    array's shape.
    """
    rough = np.round(rough).astype(int)
    surroundings = _get_subarr(array, rough, rad)
    com = _argmax_ext(surroundings, 1)
    offset = com - rad
    ret = rough + offset
    # similar to win.wrap, so
    # -0.2 becomes 0.3 and then again -0.2, which is rounded to 0
    # -0.8 becomes - 0.3 -> len() - 0.3 and then len() - 0.8,
    # which is rounded to len() - 1. Yeah!
    ret += 0.5
    ret %= np.array(array.shape).astype(int)
    ret -= 0.5
    return ret


def _get_success(array, coord, radius=2):
    """
    Given a coord, examine the array around it and return a number signifying
    how good is the "match".

    Args:
        radius: Get the success as a sum of neighbor of coord of this radius
        coord: Coordinates of the maximum. Float numbers are allowed
            (and converted to int inside)

    Returns:
        Success as float between 0 and 1 (can get slightly higher than 1).
        The meaning of the number is loose, but the higher the better.
    """
    coord = np.round(coord).astype(int)
    coord = tuple(coord)

    subarr = _get_subarr(array, coord, 2)

    theval = subarr.sum()
    theval2 = array[coord]
    # bigval = np.percentile(array, 97)
    # success = theval / bigval
    # TODO: Think this out
    success = np.sqrt(theval * theval2)
    return success

def argmax_translation(array, filter_pcorr, constraints=None):
    if constraints is None:
        constraints = dict(tx=(0, None), ty=(0, None))

    # We want to keep the original and here is obvious that
    # it won't get changed inadvertently
    array_orig = array.copy()
    if filter_pcorr > 0:
        # array = ndi.minimum_filter(array, filter_pcorr)
        array = minimum_filter_numpy(array, filter_pcorr)

    ashape = np.array(array.shape, int)
    mask = np.ones(ashape, float)
    # first goes Y, then X
    for dim, key in enumerate(("ty", "tx")):
        if constraints.get(key, (0, None))[1] is None:
            continue
        pos, sigma = constraints[key]
        alen = ashape[dim]
        dom = np.linspace(-alen // 2, -alen // 2 + alen, alen, False)
        if sigma == 0:
            # generate a binary array closest to the position
            idx = np.argmin(np.abs(dom - pos))
            vals = np.zeros(dom.size)
            vals[idx] = 1.0
        else:
            vals = np.exp(-((dom - pos) ** 2) / sigma**2)
        if dim == 0:
            mask *= vals[:, np.newaxis]
        else:
            mask *= vals[np.newaxis, :]

    array *= mask

    # WE ARE FFTSHIFTED already.
    # ban translations that are too big
    aporad = (ashape // 6).min()
    mask2 = get_apofield(ashape, aporad)
    array *= mask2
    # Find what we look for
    tvec = _argmax_ext(array, "inf")
    tvec = _interpolate(array_orig, tvec)

    # If we use constraints or min filter,
    # array_orig[tvec] may not be the maximum
    success = _get_success(array_orig, tuple(tvec), 2)

    return tvec, success


def translation(im0, im1, filter_pcorr=0, odds=1, constraints=None):
    """
    Return translation vector to register images.
    It tells how to translate the im1 to get im0.

    Args:
        im0 (2D numpy array): The first (template) image
        im1 (2D numpy array): The second (subject) image
        filter_pcorr (int): Radius of the minimum spectrum filter
            for translation detection, use the filter when detection fails.
            Values > 3 are likely not useful.
        constraints (dict or None): Specify preference of seeked values.
            For more detailed documentation, refer to :func:`similarity`.
            The only difference is that here, only keys ``tx`` and/or ``ty``
            (i.e. both or any of them or none of them) are used.
        odds (float): The greater the odds are, the higher is the preferrence
            of the angle + 180 over the original angle. Odds of -1 are the same
            as inifinity.
            The value 1 is neutral, the converse of 2 is 1 / 2 etc.

    Returns:
        dict: Contains following keys: ``angle``, ``tvec`` (Y, X),
            and ``success``.
    """
    angle = 0
    # We estimate translation for the original image...
    tvec, succ = _phase_correlation(
        im0, im1, argmax_translation, filter_pcorr, constraints
    )
    # ... and for the 180-degrees rotated image (the rotation estimation
    # doesn't distinguish rotation of x vs x + 180deg).
    ret = np.rot90(im1, 2)  # Rotate the input array over 180°
    tvec2, succ2 = _phase_correlation(
        im0, im1, argmax_translation, filter_pcorr, constraints
    )

    pick_rotated = False
    if succ2 * odds > succ or odds == -1:
        pick_rotated = True

    if pick_rotated:
        tvec = tvec2
        succ = succ2
        angle += 180

    ret = dict(tvec=tvec, success=succ, angle=angle)
    return ret


# Resize image's largest axis, by a division_factor (ratio is kept)
def resize_image(im, division_factor):
    # Use largest axis for resizing image
    largest_axis = 0
    if im.shape[1] > im.shape[0]:
        largest_axis = 1

    multiplication_factor = (
        int(im.shape[largest_axis] / division_factor) / im.shape[largest_axis]
    )

    return cv2.resize(
        im,
        (
            math.floor(im.shape[1] * multiplication_factor),
            math.floor(im.shape[0] * multiplication_factor),
        ),
    )


class im_reg:
    # Register im1 to im0 for Translation only
    def register_image_translation(self, im0, im1, scale_factor):
        translation_result = translation(
            resize_image(cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY), scale_factor),
            resize_image(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), scale_factor),
        )

        height, width = im1.shape[:2]
        # Upscale offset, as it was calculated on a smaller image
        y_shift, x_shift = translation_result["tvec"] * scale_factor

        translation_matrix = np.float64(
            [
                [1, 0, x_shift],
                [0, 1, y_shift],
            ]
        )
        result = cv2.warpAffine(im1, translation_matrix, (width, height))

        return result.astype(np.uint8)
