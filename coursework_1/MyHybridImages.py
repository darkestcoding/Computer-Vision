import numpy as np

from MyConvolution import convolve


def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.

    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or
    colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering
    lowImage
    :type float
    
    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or
    colour shape=(rows,cols,channels))
    :type numpy.ndarray
    
    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering
    highImage before subtraction to create the high-pass filtered image
    :type float
    
    :returns returns the hybrid image created
         by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it
    with
         a high-pass image created by subtracting highImage from highImage convolved with
         a Gaussian of s.d. highSigma. The resultant image has the same size as the input
    images.
    :rtype numpy.ndarray
    """

    low_image = convolve(image=lowImage.astype(float) / 255.0, kernel=makeGaussianKernel(sigma=lowSigma))
    high_image = highImage.astype(float)/ 255.0 - convolve(image=highImage.astype(float) / 255.0, kernel=makeGaussianKernel(sigma=highSigma))
    H_l,W_l = low_image.shape[:2]
    H_h,W_h = high_image.shape[:2]
    H = H_l if H_l < H_h else H_h
    W = W_l if W_l < W_h else W_h
    if low_image.ndim == 3:
        low_image_crop = low_image[:H, :W, :]
    else:
        low_image_crop = low_image[:H, : W]
    if high_image.ndim == 3:
        high_image_crop = high_image[:H, :W, :]
    else:
        high_image_crop = high_image[:H,: W]
    return ((low_image_crop + high_image_crop) * 255.0).astype(int)



def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    size = (int)(8.0 * sigma + 1.0)
    if size % 2 == 0:
        size += 1
    num_space = np.linspace(-(size//2), size//2, size)
    x, y = np.meshgrid(num_space, num_space)
    kernel = np.exp(-(x**2 + y**2)/(2 * sigma**2))/(2 * np.pi * sigma**2)
    kernel = kernel/np.sum(kernel)
    return kernel


