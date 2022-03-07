from MyHybridImages import myHybridImages
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

if __name__ == '__main__':
    low_img = mpimg.imread("anakin.jpeg")
    high_img = mpimg.imread("vader.jpeg")
    low_img_gl = mpimg.imread("anakin_gl.jpeg")
    high_img_gl = mpimg.imread("vader_gl.jpeg")
    low_sigma = 6.0
    high_sigma = 6.0
    hybrid_img = myHybridImages(low_img, low_sigma, high_img, high_sigma)
    hybrid_img_gl = myHybridImages(low_img_gl, low_sigma, high_img_gl, high_sigma)
    plt.imshow(hybrid_img)
    plt.show()
    plt.imshow(hybrid_img_gl)
    plt.show()

