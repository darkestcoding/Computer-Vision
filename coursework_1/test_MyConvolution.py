import numpy as np
from MyConvolution import convolve
from scipy import signal

def test_function(iteration):
    for i in range(iteration):
        # generate random images and kernel

        template_size = np.random.randint(3, 15)//2*2+1
        image_size =[np.random.randint(20, 80), np.random.randint(20, 80)]

        img_rgb = np.random.rand(image_size[0], image_size[1], 3)
        img_grey = np.random.rand(image_size[0], image_size[1])
        kernel = np. random.rand(template_size, template_size)

        # my convolution function
        img_rgb_conv = convolve(img_rgb, kernel)
        img_grey_conv = convolve(img_grey, kernel)

        #scipy convolution function
        img_rgb_conv_sp = np.zeros(img_rgb.shape)
        for c in range(img_rgb.shape[2]):
            img_rgb_conv_sp[:,:,c] = signal.convolve2d(img_rgb[:,:,c], kernel, mode='same')
        img_grey_conv_sp = signal.convolve2d(img_grey, kernel, mode='same')

        # assert
        assert(img_grey_conv.all() == img_grey_conv_sp.all())
        assert(img_rgb_conv.all() == img_rgb_conv_sp.all())


def test_error():
    image_size =np.random.randint(20, 80)
    img_wrong = np.random.rand(image_size, image_size, image_size,image_size)
    template_size = np.random.randint(3, 15) // 2 * 2 + 1
    kernel = np.random.rand(template_size, template_size)
    try:
        img_rgb_conv = convolve(img_wrong, kernel)
    except SystemExit:
        assert(True)
    else:
        assert(False)



if __name__ == '__main__':
    iteration = 10
    test_function(iteration=10)
    test_error()



