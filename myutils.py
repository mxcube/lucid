from math import ceil
import numpy
def expand(input_img, sigma, mode="constant", cval=0.0):

    """Expand array a with its reflection on boundaries

@param a: 2D array
@param sigma: float or 2-tuple of floats
@param mode:"constant","nearest" or "reflect"
@param cval: filling value used for constant, 0.0 by default
"""
    s0, s1 = input_img.shape
    dtype = input_img.dtype
    if isinstance(sigma, (list, tuple)):
        k0 = int(ceil(float(sigma[0])))
        k1 = int(ceil(float(sigma[1])))
    else:
        k0 = k1 = int(ceil(float(sigma)))
    if k0 > s0 or k1 > s1:
        raise RuntimeError("Makes little sense to apply a kernel (%i,%i)larger than the image (%i,%i)" % (k0, k1, s0, s1))
    output = numpy.zeros((s0 + 2 * k0, s1 + 2 * k1), dtype=dtype) + float(cval)
    output[k0:k0 + s0, k1:k1 + s1] = input_img
    if (mode == "mirror"):
	# 4 corners
	output[s0 + k0:, s1 + k1:] = input_img[-2:-k0 - 2:-1, -2:-k1 - 2:-1]
	output[:k0, :k1] = input_img[k0 - 0:0:-1, k1 - 0:0:-1]
	output[:k0, s1 + k1:] = input_img[k0 - 0:0:-1, s1 - 2: s1 - k1 - 2:-1]
	output[s0 + k0:, :k1] = input_img[s0 - 2: s0 - k0 - 2:-1, k1 - 0:0:-1]
	# 4 sides
	output[k0:k0 + s0, :k1] = input_img[:s0, k1 - 0:0:-1]
	output[:k0, k1:k1 + s1] = input_img[k0 - 0:0:-1, :s1]
	output[-k0:, k1:s1 + k1] = input_img[-2:s0 - k0 - 2:-1, :]
	output[k0:s0 + k0, -k1:] = input_img[:, -2:s1 - k1 - 2:-1]
    elif mode == "reflect":
    # 4 corners
        output[s0 + k0:, s1 + k1:] = input_img[-1:-k0 - 1:-1, -1:-k1 - 1:-1]
        output[:k0, :k1] = input_img[k0 - 1::-1, k1 - 1::-1]
        output[:k0, s1 + k1:] = input_img[k0 - 1::-1, s1 - 1: s1 - k1 - 1:-1]
        output[s0 + k0:, :k1] = input_img[s0 - 1: s0 - k0 - 1:-1, k1 - 1::-1]
    # 4 sides
        output[k0:k0 + s0, :k1] = input_img[:s0, k1 - 1::-1]
        output[:k0, k1:k1 + s1] = input_img[k0 - 1::-1, :s1]
        output[-k0:, k1:s1 + k1] = input_img[:s0 - k0 - 1:-1, :]
        output[k0:s0 + k0, -k1:] = input_img[:, :s1 - k1 - 1:-1]
    elif mode == "nearest":
    # 4 corners
        output[s0 + k0:, s1 + k1:] = input_img[-1, -1]
        output[:k0, :k1] = input_img[0, 0]
        output[:k0, s1 + k1:] = input_img[0, -1]
        output[s0 + k0:, :k1] = input_img[-1, 0]
    # 4 sides
        output[k0:k0 + s0, :k1] = numpy.outer(input_img[:, 0], numpy.ones(k1))
        output[:k0, k1:k1 + s1] = numpy.outer(numpy.ones(k0), input_img[0, :])
        output[-k0:, k1:s1 + k1] = numpy.outer(numpy.ones(k0), input_img[-1, :])
        output[k0:s0 + k0, -k1:] = numpy.outer(input_img[:, -1], numpy.ones(k1))
    elif mode == "wrap":
        # 4 corners
        output[s0 + k0:, s1 + k1:] = input_img[:k0,:k1]
        output[:k0, :k1] = input_img[-k0:,-k1:]
        output[:k0, s1 + k1:] = input_img[-k0:,:k1]
        output[s0 + k0:, :k1] = input_img[:k0,-k1:]
        # 4 sides
        output[k0:k0 + s0, :k1] = input_img[:,-k1:]
        output[:k0, k1:k1 + s1] = input_img[-k0:,:]
        output[-k0:, k1:s1 + k1] = input_img[:k0,:]
        output[k0:s0 + k0, -k1:] = input_img[:,:k1]
    elif mode != "constant": raise RuntimeError("Unknown mode")
        
    return output
    
    
def calc_size(shape, bloc_size):
    '''
    returns the adapted size padded to the next multiple of bloc_size
    
    shape: 2-tuple
    bloc_size: int or tuple
    '''
    if "__len__" in dir(bloc_size):
        return tuple ((i + j - 1) & ~(j - 1) for i,j in zip(shape,bloc_size))    	
    else:
        return tuple ((i + bloc_size - 1) & ~(bloc_size - 1) for i in shape)
    
    

    
    
    
    
    




