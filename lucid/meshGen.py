# coding utf8
#Python libary with opencv for detection on structural biology loops


__author__ = "Etienne Francois"
__contact__ = "etienne.francois@esrf.fr"
__copyright__ = "2013, ESRF"


import opencv
import cv
import math
from toolbox import dilate,white_detect,displayCv,imgInfo2cvImage,toRadians
import numpy as np

LUCID_CENTER_PROC = 0
LUCID_RECT_BBOX = 1
LUCID_POLY_BBOX = 2

#Not available yet
#LUCID_ENTIRE_MESHING =3



#Detection of loops in image
def generate_meshing_info(imgInfo,method = LUCID_RECT_BBOX,showVisuals=False,zoom=0,virtCenter=None):

    """|
    
    :param imgInfo: Information about the input image. Two types allowed yet : Image path or Numpy array
    :type imgInfo: String or Numpy array
    :param method: Method option for the generation of meshing information
    :type method: LUCID_CENTER_PROC (for use on centering), LUCID_RECT_BBOX (for generate rectangle for meshing, return Xmin, Ymin, Xmax, Ymax), LUCID_POLY_BBOX (for generate polygon for meshing, return an array of points)
    :param showVisuals: Display for debug
    :type showVisuals: Boolean
    :param zoom: Zoom level
    :type zoom: uint
    :param virtCenter: A virtual center if reqal center use is not wanted
    :type virtCenter: Tuple (uint,uint)

    :returns: (label,x,y) result from loop_detection function

    """
   
    image=imgInfo2cvImage(imgInfo)

    image2=cv.CloneImage(image)
    image3=cv.CloneImage(image)
    cv.Zero(image3)

    #image treatment
    dilate(image,image,2)
    cv.AdaptiveThreshold(image,image2,255,cv.CV_ADAPTIVE_THRESH_MEAN_C,cv.CV_THRESH_BINARY_INV,21+50*zoom,2)   
    storage = cv.CreateMemStorage()
    contours = cv.FindContours(cv.CloneImage(image2),storage,cv.CV_RETR_EXTERNAL,cv.CV_CHAIN_APPROX_SIMPLE,(0,0))
    contours=cv.ApproxPoly(contours,storage,cv.CV_POLY_APPROX_DP,3,True);
    cv.DrawContours(image3,contours,cv.RGB(255,255,255),cv.RGB (255,255,255),2)
    #Virtual center treatment
    virtCenter = virtualCenterTreatment(image3,virtCenter)
    #Rays treatment
    store,storeInfo = findInternalPoints(image3,image,4,virtCenter)
    #Adapt rays
    ellipsecoord = fitLoop(store,storeInfo,image,virtCenter)
    imageClone = cv.CloneImage(image)
    #Choose type of box for meshing
    if method==LUCID_CENTER_PROC:
        if showVisuals:
            return getBoundingBox(method,store,storeInfo,(image.width//2,image.height//2)),imageClone,image3,store,virtCenter
        else:
            return getBoundingBox(method,store,storeInfo,(image.width//2,image.height//2)),image3
    else:
        if showVisuals:
            return getBoundingBox(method,store,storeInfo,(image.width//2,image.height//2))
        else:
            return getBoundingBox(method,store,storeInfo,(image.width//2,image.height//2))
    #polys = getPolygonBoundingBox(store,storeInfo,(image.width//2,image.height//2))


#A little very simple treatment of the virtual center
def virtualCenterTreatment(image,virtCenter):

    """|
    
    :param image: Image treated
    :type image: Opencv IplImage
    :param virtCenter: A virtual center if reqal center use is not wanted
    :type virtCenter: Tuple (uint,uint)

    :returns: Return the must use center

    """ 

    realCenter =  (image.width//2,image.height//2)
    if virtCenter ==(-1,-1):
        return realCenter
    else:
        x=virtCenter[0]
        y=virtCenter[1]
        if (x<realCenter[0]+5 and x>realCenter[0]-5) and (y<realCenter[1]+5 and y>realCenter[1]-5):
            return realCenter
        else:
            return virtCenter


def getBoundingBox(method,storage,storageInfo,center):

    """|
    
    :param method: Method use in generate_meshing_info
    :type method: LUCID_CENTER_PROC (for use on centering), LUCID_RECT_BBOX (for generate rectangle for meshing, return Xmin, Ymin, Xmax, Ymax), LUCID_POLY_BBOX (for generate polygon for meshing, return an array of points)
    :param storage: Internal points of the loop
    :type storage: Array[(uint,uint)...(uint,uint)]
    :param storageInfo: Values of constant of line equation foreach internal points. a and b for an equation ax+b.
    :type storageInfo: Array[(float,float)...(float,float)].
    :param center: Coordinate of the center of the image
    :type center: Tuple (uint,uint)

    :returns: With method LUCID_CENTER_PROC and LUCID_RECT_BBOX, return coordinates for a rectangle bounding box (xmin,ymin,xmax,ymax)
    :returns: With method LUCID_POLY_BBOX, return all points of the polygone

    """ 

    if method == LUCID_RECT_BBOX or method==LUCID_CENTER_PROC:
        listX = []
        listY = []
        for (x,y) in storage:
            listX.append(x)
            listY.append(y)
        listX.sort()
        listY.sort()
        return listX[0]-5,listY[0]-5,listX[len(listX)-1]+5,listY[len(listY)-1]+5
    elif method == LUCID_POLY_BBOX:
        i=0
        polyList =[]
        while i<len(storage):
            if storage[i][0] == center[0]:
                if storage[i][1]>center[0]:
                    polyList.append((storage[i][0],storage[i][1]-5))
                else:
                    polyList.append((storage[i][0],storage[i][1]+5))
            elif storage[i][0]>center[0]:
                polyList.append((storage[i][0]+3,int(round(((storage[i][0]+3)*storageInfo[i][0])+storageInfo[i][1]))))
            elif storage[i][0]<center[0]:
                polyList.append((storage[i][0]-3,int(round(((storage[i][0]-3)*storageInfo[i][0])+storageInfo[i][1]))))
            i+=1
        return polyList

#Find inetrnal points of the loop with arc launching method
def findInternalPoints(image,image2,pointsPerArcs,center=(-1,-1)):
    
    """|
    
    :param image: Image after first treatment
    :type image: Opencv IplImage
    :param image2: Initial image treated
    :type image2: Opencv IplImage
    :param storage: Internal points of the loop
    :type storage: Array[(uint,uint)...(uint,uint)]
    :param storageInfo: Values of constant of line equation foreach internal points. a and b for an equation ax+b.
    :type storageInfo: Array[(float,float)...(float,float)].
    :param center: Coordinate of the center of the image
    :type center: Tuple (uint,uint)

    :returns: With method LUCID_CENTER_PROC and LUCID_RECT_BBOX, return coordinates for a rectangle bounding box (xmin,ymin,xmax,ymax)
    :returns: With method LUCID_POLY_BBOX, return all points of the polygone

    """
    
    #init list for points (storage) and for curve equation information
    storage = []
    storageInfo = []
    if center == (-1,-1):
        center = (image.width//2+50,image.height//2)
    i = center[1]
    while i>=0:
        if image[i,center[0]]>0:
            storage.append((center[0],i))
            storageInfo.append((0,1))
            break;
	else:
            i=i-1
    #print storage[0][0]
    for degrees in np.arange(15,360,15):
    #for degrees in np.arange(5,360,5):
	    x2 = storage[0][0]-center[0]
	    y2 = storage[0][1]-center[1]

	    xrot = x2*math.cos(toRadians(degrees))-y2*math.sin(toRadians(degrees))
	    yrot = y2*math.cos(toRadians(degrees))+x2*math.sin(toRadians(degrees))

	    xrot = round(xrot + center[0])
	    yrot = round(yrot + center[1])
            #cv.Line( image2, (image.width//2,image.height//2), (xrot,yrot),cv.Scalar( 60, 60,60 ),2,4 );
	    #print "inter"
            if center[0]==xrot:
	            m=1
	            p=0
                    i = center[1]
		    while i<image.height:
			if image[i,center[0]]>0:
			    storage.append((center[0],i))
                            storageInfo.append((m,p))
			    break;
			else:
			    i=i+1
                        if i==image.height:
                            storage.append((-1,-1))
            else:
		    m = (yrot - center[1])/(xrot - center[0])
		    p = -(m*xrot)+yrot
		    #print center[0],center[1],xrot,yrot,m,p
		    x = center[0]
		    y = center[1]
		    ok =False

		    while 1:
                        if degrees<190:
			    x=x+1
                        else:
                            x=x-1
			y=m*x+p
			y = int(round(y))
			if x<image.width-2 and x>=2 and y<image.height-2 and y>=2:
			    if white_detect(image[(y-2):(y+2),(x-2):(x+2)])>0:
				storage.append((x,y))
                                storageInfo.append((m,p))
				ok = True
				#cv.Circle( image,storage[0],5,cv.RGB(255,255,255 ),-1,8 )
				break
			else:
			    break
		    if ok == False:
			    x = center[0]
			    y = center[1]
			    while 1:
		                if degrees<90 or degrees>270:
				    y=y-1
		                else:
		                    y=y+1
                                if m==0:
                                   #storage.append((0,0))
                                   break
				x=(y-p)/m
				x = int(round(x))
				if x<image.width-2 and x>=2 and y<image.height-2 and y>=2:
				    if white_detect(image[(y-2):(y+2),(x-2):(x+2)])>0:
					storage.append((x,y))
                                        storageInfo.append((m,p))
					#cv.Circle( image,storage[0],5,cv.RGB(255,255,255 ),-1,8 )
					break
				else:
				    #storage.append((0,0))
				    break
    return storage,storageInfo

def fitLoop(storage,storageInfo,image,virtCenter):

    center = virtCenter
    res = 0
    
    #Average of the distance to the center
    for num in range(len(storage)):
        ptb = storage[num]
        res += math.sqrt(((ptb[0]-center[0])*(ptb[0]-center[0]))+((ptb[1]-center[1])*(ptb[1]-center[1])))

    res = res/len(storage)
    stDev = 0
    
    #Calculating the standard deviation (in French : Ecart-type)
    for num in range(len(storage)):
        ptb = storage[num]
        stDev += (math.sqrt(((ptb[0]-center[0])*(ptb[0]-center[0]))+((ptb[1]-center[1])*(ptb[1]-center[1]))) - res)*(math.sqrt(((ptb[0]-center[0])*(ptb[0]-center[0]))+((ptb[1]-center[1])*(ptb[1]-center[1]))) - res)

    stDev = stDev/(len(storage)-1)
    stDev = math.sqrt(stDev)

    #print "moyenne"
    #print res
    #print "ecart type"
    #print stDev
    #print "somme"
    #print stDev+res

    #Define a limit : Average + Standard Deviation
    limit = res+stDev

    #Define a list of rays with a distance to the center greater or equals than the limit
    warningList = []
    for num in range(len(storage)):
        ptb = storage[num]
        res = math.sqrt(((ptb[0]-center[0])*(ptb[0]-center[0]))+((ptb[1]-center[1])*(ptb[1]-center[1])))
        if res >= limit:
             warningList.append(num)
    warningList.sort()

    #Check if rays detected are really non consistant
    while len(warningList) >= 1:
        boolR = checkConsistancy(image,storage,storageInfo,warningList,center,warningList[0],'r')
        del(warningList[warningList.index(warningList[0])])    
    #print 'achtungicht',list(set(warningList))
    '''
    PointArray = cv.CreateMat(1,len(storage),cv.CV_32FC2)
    for num in range(len(storage)):
        PointArray[0,num] = storage[num]
    return cv.FitEllipse2(PointArray)
    '''

def checkConsistancy(image,storage,storageInfo,warningList,center,iden,side):
    #print warningList,iden
    i = 0

    #Calcultate distance to center
    ptb = storage[iden]
    res = math.sqrt(((ptb[0]-center[0])*(ptb[0]-center[0]))+((ptb[1]-center[1])*(ptb[1]-center[1])))
    ptb = storage[(iden - 1)%len(storage)]
    res2 = math.sqrt(((ptb[0]-center[0])*(ptb[0]-center[0]))+((ptb[1]-center[1])*(ptb[1]-center[1])))
    ptb = storage[(iden + 1)%len(storage)]
    res3 = math.sqrt(((ptb[0]-center[0])*(ptb[0]-center[0]))+((ptb[1]-center[1])*(ptb[1]-center[1])))

    #Compare to know if a modification is needed
    if res>res2+(0.6*res2) or res>res3+(0.6*res3):
        ptb = storage[iden]
        m = storageInfo[iden][0]
        p = storageInfo[iden][1]
           
        #Find a reference after the studied point
        indexL = (iden + 1)%len(storage)
        ecartL = 1
        lengthPropL=0

        while 1:
             #Check if the first ref after is or is not in the list of warning rays, if not pass the the next loop's iteration
             if (indexL in warningList):
                 indexL = (indexL + 1)%len(storage)
                 ecartL += 1
             #Check if the ref after the first ref is not in list and define a length with these two ref
             elif (indexL + 1)%len(storage) not in warningList:
                 #Explain TODO
                 lengthPropL = math.sqrt(((storage[indexL][0]-center[0])*(storage[indexL][0]-center[0]))+((storage[indexL][1]-center[1])*(storage[indexL][1]-center[1]))) + abs((math.sqrt(((storage[indexL][0]-center[0])*(storage[indexL][0]-center[0]))+((storage[indexL][1]-center[1])*(storage[indexL][1]-center[1]))))-(math.sqrt(((storage[(indexL + 1)%len(storage)][0]-center[0])*(storage[(indexL + 1)%len(storage)][0]-center[0]))+((storage[(indexL + 1)%len(storage)][1]-center[1])*(storage[(indexL + 1)%len(storage)][1]-center[1])))))
                 break
             #if the first ref is ok but not the second one, define a length based only on the first one
             else:
                 lengthPropL = math.sqrt(((storage[indexL][0]-center[0])*(storage[indexL][0]-center[0]))+((storage[indexL][1]-center[1])*(storage[indexL][1]-center[1])))
                 break

        #Explanation identical to the precedent one
        indexR =(iden - 1)%len(storage)
        ecartR = 1
        lengthPropR=0
        while 1:
            if (indexR in warningList):
                 indexR = (indexR - 1)%len(storage)
                 ecartR += 1
            elif (indexR - 1)%len(storage) not in warningList:
                 lengthPropL = math.sqrt(((storage[indexR][0]-center[0])*(storage[indexR][0]-center[0]))+((storage[indexR][1]-center[1])*(storage[indexR][1]-center[1]))) + abs((math.sqrt(((storage[indexR][0]-center[0])*(storage[indexR][0]-center[0]))+((storage[indexR][1]-center[1])*(storage[indexR][1]-center[1]))))-(math.sqrt(((storage[(indexR-1)%len(storage)][0]-center[0])*(storage[(indexR-1)%len(storage)][0]-center[0]))+((storage[(indexR-1)%len(storage)][1]-center[1])*(storage[(indexR-1)%len(storage)][1]-center[1])))))
                 break
            else:
                 lengthPropR = math.sqrt(((storage[indexR][0]-center[0])*(storage[indexR][0]-center[0]))+((storage[indexR][1]-center[1])*(storage[indexR][1]-center[1])))
                 break


        #print (((ecartR*lengthPropL)+(ecartL*lengthPropR))/(ecartR+ecartL))             
        x=center[0]
        while 1:
            if center[0]<ptb[0]:
                x=x+1
            else:
                x=x-1
	    y=m*x+p
            y = int(round(y))
	    if x<image.width-2 and x>0 and y<image.height-2 and y>0:
                l = math.sqrt(((x-center[0])*(x-center[0]))+((y-center[1])*(y-center[1])))
                    
	        if l>=(((ecartR*lengthPropL)+(ecartL*lengthPropR))/(ecartR+ecartL)):
	    	    #print l
                    storage[iden] = (x,y)
                    #print (x,y),storage
                    break
            else:
		break
        return True
    else:
        return False
