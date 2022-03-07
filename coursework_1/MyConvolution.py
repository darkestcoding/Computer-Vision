import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) ->np.ndarray:
    """

    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type  numpy.ndarray

    :param kernel:the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray

    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray

    """

    W = image.shape[0]
    H = image.shape[1]
    if image.ndim == 3:
        channel_num = 3
        img_new = np.ndarray((W, H, channel_num))
    elif image.ndim == 2:
        channel_num = 1
        img_new = np.ndarray((W, H))
    else:
        raise SystemExit('please input correct RGB or grey-scale image!')


    # padding based on different channels
    pad_size = kernel.shape[0]//2
    if channel_num == 3:
        img_padding= np.zeros((image.shape[0] + pad_size * 2, image.shape[1] + pad_size * 2, 3))
        img_padding[pad_size:img_padding.shape[0] - pad_size, pad_size:img_padding.shape[1] - pad_size, :] = image
    if channel_num == 1:
        img_padding = np.zeros((image.shape[0] + pad_size * 2, image.shape[1] + pad_size * 2))
        img_padding[pad_size:img_padding.shape[0] - pad_size, pad_size:img_padding.shape[1] - pad_size] = image

    # flip the kernel
    kernel = kernel[:, ::-1][::-1, :]

    # convolve
    for channel in range(channel_num):
        if channel_num == 1:
            for y in range(H):
                for x in range(W):
                        conv_num = np.sum(kernel * img_padding[x:x+kernel.shape[0], y:y+kernel.shape[1]])
                        img_new[x, y] = conv_num
        else:
            for y in range(H):
                for x in range(W):
                    conv_num = np.sum(kernel * img_padding[x:x + kernel.shape[0], y:y + kernel.shape[1], channel])
                    img_new[x, y, channel] = conv_num

    return img_new

