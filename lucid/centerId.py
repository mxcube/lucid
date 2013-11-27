# coding utf8
# Python libary with opencv for detection on structural biology loops


__author__ = "Etienne Francois"
__contact__ = "etienne.francois:esrf.fr"
__copyright__ = "2013, ESRF"


import opencv
import cv
import numpy
#import pyFAI.splitBBox
import pylab
import scipy.ndimage
from toolbox import displayCv,white_detect,erode,dilate,open_image,smooth,imgInfo2cvImage
import meshGen


#================================================================================#
#          									 #
#                      Pretreatment for loop centering                           #
#   										 #
#================================================================================#


# Explicit launcher for face finding with pretreatment of the image and the face detection
def find_face(imgInfo,showVisuals=False,zoom=0):
    """|

    :param imgInfo: Information about the input image. Two types allowed yet : Image path or Numpy array
    :type imgInfo: String or Numpy array
    :param showVisuals: Display for debug
    :type showVisuals: Boolean
    :param zoom: Zoom level
    :type zoom: uint

    :returns: Return from loop detection.

    """
    return find_loop(imgInfo,showVisuals=showVisuals,zoom=zoom,faceFindProc=True)

#Detection of loop with the meshing system
def find_loop_mesh(imgInfo, showVisuals=False, zoom=0, virtCenter=(-1,-1)):
    """|
    
    :param imgInfo: Information about the input image. Two types allowed yet : Image path or Numpy array
    :type imgInfo: String or Numpy array
    :param showVisuals: Display for debug
    :type showVisuals: Boolean
    :param zoom: Zoom level
    :type zoom: uint
    :param virtCenter: A virtual center if reqal center use is not wanted
    :type virtCenter: Tuple (uint,uint)

    :returns: (label,x,y) result from loop_detection function

    """
    if showVisuals:
        (x,y,x2,y2),imageClone,image3,store,virtCenter = meshGen.generate_meshing_info(imgInfo,method=meshGen.LUCID_CENTER_PROC,showVisuals=showVisuals,zoom=zoom,virtCenter=virtCenter)
        cv.Rectangle(imageClone,(x,y),(x2,y2),cv.Scalar( 120, 120, 120 ))
    else:
        (x,y,x2,y2),image3 = meshGen.generate_meshing_info(imgInfo,method=meshGen.LUCID_CENTER_PROC,showVisuals=showVisuals,zoom=zoom,virtCenter=virtCenter)
   
    xres = x2-((x2-x)//2)
    yrestemp = -1
    yrestemp2 = -1
    yres = 0
    while y<y2 and (yrestemp < 0 or yrestemp2 < 0):
        y+=1
        y2-=1
        if image3[y,xres]>0:
            yrestemp = y
        if image3[y2,xres]>0:
            yrestemp2 = y2
    if y>=y2:
        yres = y2-((y2-y)//2)
    else:
        yres = yrestemp2 - ((yrestemp2-yrestemp)//2)
    if showVisuals:
        for num in range(len(store)):
             cv.Line( image3, virtCenter, store[num],cv.Scalar( 120, 120, 120 ),2,8 );
	cv.Line( imageClone, (xres-3,yres-3), (xres+3,yres+3),cv.Scalar( 120, 120, 120 ),2,8 );
	cv.Line( imageClone, (xres-3,yres+3), (xres+3,yres-3),cv.Scalar( 120, 120, 120 ),2,8 );

        cv.Line( imageClone, (virtCenter[0],virtCenter[1]-3), (virtCenter[0],virtCenter[1]+3),cv.Scalar( 120, 120, 120 ),2,8 );
	cv.Line( imageClone, (virtCenter[0]-3,virtCenter[1]), (virtCenter[0]+3,virtCenter[1]),cv.Scalar( 120, 120, 120 ),2,8 );
        #real center
        cv.Line( imageClone, ((659//2),(463//2)-15), ((659//2),(463//2)+15),cv.Scalar( 120, 120, 120 ),2,8 );
        cv.Line( imageClone, ((659//2)-15,(463//2)), ((659//2)+15,(463//2)),cv.Scalar( 120, 120, 120 ),2,8 );

        displayCv(('Resultat',imageClone),('RaysVisu',image3))
	cv.WaitKey(0)
    return "meshing",xres,yres
