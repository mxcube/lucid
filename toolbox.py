# coding utf8
#Python libary with opencv for detection on structural biology loops


__author__ = "Etienne Francois"
__contact__ = "etienne.francois:esrf.fr"
__copyright__ = "2013, ESRF"


#import cv
import scipy
import os
import Image
import numpy
import math


#================================================================================#
#          									 #
#                    Python, python libraries, os functions                      #
#   										 #
#================================================================================#


#Open an image from disk
def open_image(filename):
    return Image.open(os.path.abspath(os.path.join(os.path.dirname(__file__), filename)))

#Convert degrees to radian
def toRadians(degrees):
    return degrees * ((2 * math.pi) / 360)


#================================================================================#
#          									 #
#                           OpenCv linked functions                              #
#   										 #
#================================================================================#


#Convert disponible image information into an IplImage
def imgInfo2cvImage(imgInfo):
    """|

    :param imgInfo: Image or part of image
    :type imgInfo: String or Numpy array

    :returns: OpenCV IplImage 
    """
    #Test what type of image data has been given as argument "imgInfo"
    if isinstance(imgInfo,numpy.ndarray):
        return cv.GetImage(cv.fromarray(imgInfo))
    elif isinstance(imgInfo,str):
        return cv.LoadImage(imgInfo,cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        raise TypeError("Unsupported type : Image path (str) or numpyarray (numpy.ndarray) needed") 

#Display fonction for cvImage
def displayCv(*tupleDisplayable):
    """|

    :param *tupleDisplayable: Displayable tuples
    :type *tupleDisplayable: Multiples arguments : Tuples : (\"Window's name\",[cvImage]) 
    """
    for tp in list(tupleDisplayable):
        if isinstance(tp,tuple):
            try:
                cv.NamedWindow(tp[0], cv.CV_WINDOW_AUTOSIZE)
                cv.ShowImage(tp[0],tp[1])
                continue
            except Exception:
                print "An error occurs on displaying cv visuals. Please check your tuples in input. Format : (\"WindowName\",CvImage)"
                return None

#Detection of white pixel in a binary image (0/255)
def white_detect(cvImg):
    """|

    :param cvImg: Image or part of image
    :type cvImg: OpenCV IplImage or part of OpenCV IplImage (image[:,0:150] for example)

    :returns: The numbers of white pixels in b/w image
    """
    return (cv.Sum(cvImg))[0]/255

#Morphological mathematic erosion function with num for how many times the file might be treated
def erode(src,dest,num):
    """|
    
    :param src : Input image
    :type src : IplImage
    :param dest : Output image
    :type dest : IplImage
    :param num : How much time erosion have to be applied >0
    :type num : uint
    """
    i=0
    fich = cv.CloneImage(src)
    while i<num-1:
        cv.Erode(fich,fich)
        i = i+1
    cv.Erode(fich,dest)

#Morphological mathematic dilatation function with num for how many times the file might be treated
def dilate(src,dest,num):
    """|
    
    :param src : Input image
    :type src : IplImage
    :param dest : Output image
    :type dest : IplImage
    :param num : How much time dilatation have to be applied >0
    :type num : uint
    """
    i=0
    fich = cv.CloneImage(src)
    while i<num-1:
        cv.Dilate(fich,fich)
        i = i+1
    cv.Dilate(fich,dest)

#Smooth application on image with num for how many times the file might be treated
def smooth(src,dest,num):
    """|
    
    :param _src: Input image
    :type _src: IplImage
    :param _dest: Output image
    :type _dest: IplImage
    :param _num: How much time gaussian blur have to be applied >0
    :type _num: uint

    :returns: Output image
    """
    i=0
    fich = cv.CloneImage(src)
    while i<num-1:
        cv.Smooth(fich,fich)
        i = i+1
    cv.Smooth(fich,dest)
    return dest


def threshold_adaptive(image, block_size, method='gaussian', offset=0,
                       mode='reflect', param=None):
    """Applies an adaptive threshold to an array.

Also known as local or dynamic thresholding where the threshold value is
the weighted mean for the local neighborhood of a pixel subtracted by a
constant. Alternatively the threshold can be determined dynamically by a a
given function using the 'generic' method.

Parameters
----------
image : (N, M) ndarray
Input image.
block_size : int
Uneven size of pixel neighborhood which is used to calculate the
threshold value (e.g. 3, 5, 7, ..., 21, ...).
method : {'generic', 'gaussian', 'mean', 'median'}, optional
Method used to determine adaptive threshold for local neighbourhood in
weighted mean image.

* 'generic': use custom function (see `param` parameter)
* 'gaussian': apply gaussian filter (see `param` parameter for custom\
sigma value)
* 'mean': apply arithmetic mean filter
* 'median': apply median rank filter

By default the 'gaussian' method is used.
offset : float, optional
Constant subtracted from weighted mean of neighborhood to calculate
the local threshold value. Default offset is 0.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
The mode parameter determines how the array borders are handled, where
cval is the value when mode is equal to 'constant'.
Default is 'reflect'.
param : {int, function}, optional
Either specify sigma for 'gaussian' method or function object for
'generic' method. This functions takes the flat array of local
neighbourhood as a single argument and returns the calculated
threshold for the centre pixel.

Returns
-------
threshold : (N, M) ndarray
Thresholded binary image

References
----------
.. [1] http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#adaptivethreshold

Examples
--------
>>> from skimage.data import camera
>>> image = camera()
>>> binary_image1 = threshold_adaptive(image, 15, 'mean')
>>> func = lambda arr: arr.mean()
>>> binary_image2 = threshold_adaptive(image, 15, 'generic', param=func)
"""
    thresh_image = numpy.zeros(image.shape, 'double')
    if method == 'generic':
        scipy.ndimage.generic_filter(image, param, block_size,
            output=thresh_image, mode=mode)
    elif method == 'gaussian':
        if param is None:
            # automatically determine sigma which covers > 99% of distribution
            sigma = (block_size - 1) / 6.0
        else:
            sigma = param
        scipy.ndimage.gaussian_filter(image, sigma, output=thresh_image,
            mode=mode)
    elif method == 'mean':
        mask = 1. / block_size * numpy.ones((block_size,))
        # separation of filters to speedup convolution
        scipy.ndimage.convolve1d(image, mask, axis=0, output=thresh_image,
            mode=mode)
        scipy.ndimage.convolve1d(thresh_image, mask, axis=1,
            output=thresh_image, mode=mode)
    elif method == 'median':
        scipy.ndimage.median_filter(image, block_size, output=thresh_image,
            mode=mode)

    return image > (thresh_image - offset)

def get_background_threshold(image, mask = None):
    """Get threshold based on the mode of the image
    The threshold is calculated by calculating the mode and multiplying by
    2 (an arbitrary empirical factor). The user will presumably adjust the
    multiplication factor as needed."""
    cropped_image = numpy.array(image.flat) if mask is None else image[mask]
    if numpy.product(cropped_image.shape)==0:
        return 0
    img_min = numpy.min(cropped_image)
    img_max = numpy.max(cropped_image)
    if img_min == img_max:
        return cropped_image[0]
    
    # Only do the histogram between values a bit removed from saturation
    robust_min = 0.02 * (img_max - img_min) + img_min
    robust_max = 0.98 * (img_max - img_min) + img_min
    nbins = 256
    cropped_image = cropped_image[numpy.logical_and(cropped_image > robust_min,
                                                 cropped_image < robust_max)]
    if len(cropped_image) == 0:
        return robust_min
    
    h = scipy.ndimage.histogram(cropped_image, robust_min, robust_max, nbins)
    index = numpy.argmax(h)
    cutoff = float(index) / float(nbins-1)
    #
    # If we have a low (or almost no) background, the cutoff will be
    # zero since the background falls into the lowest bin. We want to
    # offset by the robust cutoff factor of .02. We rescale by 1.04
    # to account for the 0.02 at the top and bottom.
    #
    cutoff = (cutoff + 0.02) / 1.04
    return img_min + cutoff * 2 * (img_max - img_min)

def get_robust_background_threshold(image, mask = None):
    """Calculate threshold based on mean & standard deviation
       The threshold is calculated by trimming the top and bottom 5% of
       pixels off the image, then calculating the mean and standard deviation
       of the remaining image. The threshold is then set at 2 (empirical
       value) standard deviations above the mean.""" 

    cropped_image = numpy.array(image.flat) if mask is None else image[mask]
    if numpy.product(cropped_image.shape)<3:
        return 0
    if numpy.min(cropped_image) == numpy.max(cropped_image):
        return cropped_image[0]
    
    cropped_image.sort()
    chop = int(round(numpy.product(cropped_image.shape) * .05))
    im   = cropped_image[chop:-chop]
    mean = im.mean()
    sd   = im.std()
    return mean+sd*2


def hist_eq(im,nbr_bins=256):
   imhist,bins = numpy.histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = numpy.interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape), cdf
