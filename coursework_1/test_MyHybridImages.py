import MyHybridImages
import cv2
import os


def test_makeGaussianKernel(sigma):
    kernel_x = cv2.getGaussianKernel(ksize=(int)(8.0 * sigma + 1.0), sigma=sigma)
    kernel_y = cv2.getGaussianKernel(ksize=(int)(8.0 * sigma + 1.0), sigma=sigma)
    kernel_cv2 = kernel_x @ kernel_y.T
    kernel_my = MyHybridImages.makeGaussianKernel(sigma=sigma)
    assert (kernel_my.all() == kernel_cv2.all())


def test_myHybridImages(high_image,low_image,high_sigma,low_sigma):
    hybrid_image = MyHybridImages.myHybridImages(highImage=high_image,lowImage=low_image,
                                                 highSigma=high_sigma,lowSigma=low_sigma)
    cv2.imwrite("hybrid_image.jpeg", hybrid_image)
    assert os.path.exists("hybrid_image.jpeg")


if __name__ == '__main__':
    kernel = test_makeGaussianKernel(sigma=1.0)

    low_img_gl = cv2.imread("anakin_gl.jpeg")
    high_img_gl = cv2.imread("vader_gl.jpeg")
    low_sigma = 6.0
    high_sigma = 6.0
    test_myHybridImages(high_image=high_img_gl,low_image=low_img_gl,
                                       high_sigma=high_sigma,low_sigma=low_sigma)