import numpy as np
from PIL import Image
import math

############### ---------- Basic Image Processing ------ ##############

### TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image
def imread(filename):
    img = Image.open(filename)
    img = img.convert("RGB")
    arr = np.array(img)
    arr = arr.astype(np.float32) / 255.0
    return arr


### TODO 2: Convolve an image (m x n x 3 or m x n) with a filter(l x k). Perform "same" filtering. Apply the filter to each channel if there are more than 1 channels
def convolve(img, filt):
    filter = np.fliplr(np.flipud(filt))
    if len(img.shape) == 2: img = np.expand_dims(img, 2)

    height, width, channels = img.shape
    filt_h, filt_w = filter.shape

    padded = np.pad(
        img,
        [
            ((filt_h - 1) // 2, (filt_h - 1) // 2),
            ((filt_w - 1) // 2, (filt_w - 1) // 2),
            (0, 0),
        ],
        "constant",
    )
    filtered = np.zeros((height, width, channels))

    for i in range(height):
        for j in range(width):
            for k in range(channels):
                filtered[i, j, k] = (
                    filter * padded[i : i + filt_h, j : j + filt_w, k]
                ).sum()

    if channels == 1:
        filtered = np.reshape(filtered, (height, width))

    return filtered


### TODO 3: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):
    gauss_filter = np.zeros((k, k))
    center = k // 2
    for x in range(-center, center + 1):
        for y in range(-center, center + 1):
            gauss_filter[x + center, y + center] = np.exp(
                -(x**2 + y**2) / (2 * sigma**2)
            )
    gauss_filter = gauss_filter / gauss_filter.sum()
    return gauss_filter


### TODO 4: Compute the image gradient.
### First convert the image to grayscale by using the formula:
### Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
### Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise.
### Convolve with [0.5, 0, -0.5] to get the X derivative on each channel
### convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel
### Return the gradient magnitude and the gradient orientation (use arctan2)
def gradient(img):
    intensity = img[:, :, 0] * 0.2125 + img[:, :, 1] * 0.7154 + img[:, :, 2] * 0.0721

    smoothed = convolve(intensity, gaussian_filter(5, 1))

    xd = convolve(smoothed, np.array([[0.5, 0, -0.5]]))
    yd = convolve(smoothed, np.array([[0.5], [0], [-0.5]]))

    gradmag = np.sqrt(xd**2 + yd**2)
    gradori = np.arctan2(yd, xd)

    return gradmag, gradori


##########----------------Line detection----------------

### TODO 5: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
### x cos(theta) + y sin(theta) + c = 0
### The input x and y are numpy arrays of the same shape, representing the x and y coordinates of each pixel
### Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):
    x = np.array(x)
    y = np.array(y)
    distance = np.abs(x * np.cos(theta) + y * np.sin(theta) + c) / np.sqrt(
        np.cos(theta) ** 2 + np.sin(theta) ** 2
    )
    out = distance < thresh
    return out


### TODO 6: Write a function to draw a set of lines on the image.
### The `img` input is a numpy array of shape (m x n x 3).
### The `lines` input is a list of (theta, c) pairs.
### Mark the pixels that are less than `thresh` units away from the line with red color,
### and return a copy of the `img` with lines.
def draw_lines(img, lines, thresh):
    img = np.copy(img)
    height, width, _ = img.shape
    for theta, c in lines:
        for x, y in np.ndindex(height, width):
            if check_distance_from_line([x], [y], theta, c, thresh):
                img[y, x, :] = [1, 0, 0]
    return img


### TODO 7: Do Hough voting. You get as input the gradient magnitude (m x n) and the gradient orientation (m x n),
### as well as a set of possible theta values and a set of possible c values.
### If there are T entries in thetas and C entries in cs, the output should be a T x C array.
### Each pixel in the image should vote for (theta, c) if:
### (a) Its gradient magnitude is greater than thresh1, **and**
### (b) Its distance from the (theta, c) line is less than thresh2, **and**
### (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    out = np.zeros((len(thetas), len(cs)))
    height, width = gradmag.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    arr = np.where(gradmag > thresh1, True, False)
    for i, theta in enumerate(thetas):
        for j, c in enumerate(cs):
            distances = check_distance_from_line(x, y, theta, c, thresh2)
            arr3 = np.where(np.abs(theta - gradori) < thresh3, True, False)
            out[i, j] = np.sum(arr * distances * arr3)
    return out


### TODO 8: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if:
### (a) Its votes are greater than thresh, **and**
### (b) Its value is the maximum in a nbhd x nbhd beighborhood in the votes array.
### The input `nbhd` is an odd integer, and the nbhd x nbhd neighborhood is defined with the
### coordinate of the potential local maxima placing at the center.
### Return a list of (theta, c) pairs.
def localmax(votes, thetas, cs, thresh, nbhd):
    out = []
    thresh_points = np.argwhere(votes > thresh)
    padding = nbhd // 2
    padded = np.pad(votes, ((padding, padding), (padding, padding)), "constant")
    for theta, c in thresh_points:
        vote_count = votes[theta, c]
        flag = True
        for i in range(theta - nbhd, theta + nbhd):
            if not flag:
                break
            for j in range(c - padding, c + padding + 1):
                if vote_count < padded[i, j]:
                    flag = False
                    break
        if flag:
            out.append((thetas[theta], cs[c]))
    return out


# Final product: Identify lines using the Hough transform
def do_hough_lines(filename):

    # Read image in
    img = imread(filename)

    # Compute gradient
    gradmag, gradori = gradient(img)

    # Possible theta and c values
    thetas = np.arange(-np.pi - np.pi / 40, np.pi + np.pi / 40, np.pi / 40)
    imgdiagonal = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    cs = np.arange(-imgdiagonal, imgdiagonal, 0.5)

    # Perform Hough voting
    votes = hough_voting(gradmag, gradori, thetas, cs, 0.1, 0.5, np.pi / 40)

    # Identify local maxima to get lines
    lines = localmax(votes, thetas, cs, 20, 11)

    # Visualize: draw lines on image
    result_img = draw_lines(img, lines, 0.5)

    # Return visualization and lines
    return result_img, lines
