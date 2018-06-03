import math
import PIL.Image as Image
import torchvision.transforms.functional as F


def crop_square(im, r, c, sz):
    '''
    crop image into a square of size sz,
    '''
    return im[r:r+sz, c:c+sz]


def crop(im, r, c, sz_h, sz_w):
    '''
    crop image into a square of size sz,
    '''
    return im[r:r+sz_h, c:c+sz_w]


def center_crop(im, min_sz=None):
    """ Returns a center crop of an image"""
    return F.center_crop(im, min_sz)
    # r,c,*_ = im.shape
    # if min_sz is None: min_sz = min(r,c)
    # start_r = math.ceil((r-min_sz)/2)
    # start_c = math.ceil((c-min_sz)/2)
    # return crop_square(im, start_r, start_c, min_sz)


def scale_to(x, ratio, targ):
    '''Calculate dimension of an image during scaling with aspect ratio'''
    return max(math.floor(x*ratio), targ)


def scale_min(im, targ, interpolation=Image.BILINEAR):
    """ Scales the image so that the smallest axis is of size targ.

    Arguments:
        im (array): image
        targ (int): target size
    """
    # r,c,*_ = im.shape
    r, c = im.size

    ratio = targ/min(r, c)

    sz = (scale_to(r, ratio, targ), scale_to(c, ratio, targ))

    return im.resize(sz, resample=interpolation)
