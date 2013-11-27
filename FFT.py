import os
import sys
import numpy
import pylab
import scipy.ndimage
import myutils as utils
import scipy.stats
import mahotas.polygon
import toolbox
import types
import math

LOOP_MAX_WIDTH = 400*1E-3 #400 microns
PIXELS_PER_MM_HOR = 320

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def img2float(fn):
    im = scipy.misc.imread(fn)
    return rgb2gray(im) 

def show_img(img,show=True, save=False, xy=None):
    #if show:
    #  pylab.imshow(img)
    #  pylab.gray()
    #  pylab.show()
    filename = os.tempnam()+".png"
    pylab.imsave(filename, img, cmap=pylab.cm.Greys_r)
    if xy is not None:
      x0 = xy[0]-5
      y0 = xy[1]-5
      x1 = x0+5
      y1 = y0+5
      os.system("convert %s -fill red -draw 'rectangle %d,%d,%d,%d' %s_" % (filename, x0, y0, x1, y1, filename)) 
      os.system("mv %s_ %s" % (filename, filename)) 
    if show:
      os.system("display %s" % filename)
    if not save:
      os.unlink(filename)
    else:
      print filename

def find_loop(img, debug=False, pixels_per_mm_horizontal=PIXELS_PER_MM_HOR):
    if type(img) == types.StringType:
        raw_img = img2float(img)
    else:
        # already a numpy array
        raw_img = img

    im = scipy.ndimage.gaussian_filter(raw_img, 1)

    if debug:
        show_img(im)

    im = im[1:-1,1:-2]
    im = utils.expand(im,2,mode="mirror")

    mask = new_mask(im.shape, .7, .7, 5)

    i1f = numpy.fft.fft2(im)
    i2f = numpy.multiply(i1f,mask)
    res = numpy.fft.ifft2(i2f)

    imgnumpy = im - res.real
    if debug:
        show_img(imgnumpy)

    dx = scipy.ndimage.sobel(imgnumpy,axis=0,mode='constant')  # horizontal derivative
    dy = scipy.ndimage.sobel(imgnumpy,axis=1,mode='constant')  # vertical derivative
    mag = numpy.hypot(dx,dy)  # magnitude
    mag *= 255.0/numpy.max(mag)
    if debug:
        show_img(mag)
    imgnumpy = numpy.uint8(mag)

    for offset in [20,15,10,5,0]:
        T = toolbox.get_robust_background_threshold(imgnumpy)
        thresholded = imgnumpy > (T+offset) 
        if debug:
            show_img(thresholded)

        binimg = scipy.ndimage.morphology.binary_opening(thresholded, iterations=2)
        
        if debug:
            show_img(binimg)

        if numpy.sum(binimg) > 50:
          break
    else:
        return ("No loop detected", -1, -1)

    min1, max1, min2, max2 = mahotas.bbox(binimg)
    if debug:
        show_img(binimg)
    binimg2 = numpy.zeros_like(binimg)
    # remove 200 microns from right
    loop_max_width_pixels = LOOP_MAX_WIDTH*pixels_per_mm_horizontal
    binimg2[min1:max1, max2-loop_max_width_pixels:max2]=binimg[min1:max1, max2-loop_max_width_pixels:max2]
    if debug:
        show_img(binimg2)
    
    min1, max1, min2, max2 = mahotas.bbox(binimg2)
    bounded_img = binimg2[min1:max1, min2:max2]   
    hull = mahotas.polygon.fill_convexhull(bounded_img)
    cy, cx = map(int, scipy.ndimage.measurements.center_of_mass(hull))
    x = int(min2)+cx
    y = int(min1)+cy
    if debug: 
        show_img(binimg2, xy=(x,y))
    
    return ("Coord",x,y)

def new_mask(shape,sigma,sigma2,mulsigma):
    h = numpy.zeros(shape[0])
    w = numpy.zeros(shape[1])

    h[(shape[0]//2)-(mulsigma*sigma):(shape[0]//2)+(mulsigma*sigma)]=1
    w[(shape[1]//2)-(mulsigma*sigma2):(shape[1]//2)+(mulsigma*sigma2)]=1

    b1 = scipy.ndimage.filters.gaussian_filter(h,sigma)
    b2 = scipy.ndimage.filters.gaussian_filter(w,sigma2)

    h0 = numpy.empty_like(b1)
    h1 = numpy.empty_like(b2)

    h0[:shape[0] // 2] = b1[shape[0] - shape[0] // 2:]
    h0[shape[0] // 2:] = b1[:shape[0] - shape[0] // 2]
    h1[:shape[1] // 2] = b2[shape[1] - shape[1] // 2:]
    h1[shape[1] // 2:] = b2[:shape[1] - shape[1] // 2]

    g = numpy.outer(h0,h1)
    return g

if __name__ == '__main__':
    for filename in sys.argv[1:]:
        print find_loop(filename, debug=False) #True)

