import cv2
import math
import numpy as np


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
    # return F.center_crop(im, min_sz)
    r,c,*_ = im.shape
    if min_sz is None: min_sz = min(r,c)
    start_r = math.ceil((r-min_sz)/2)
    start_c = math.ceil((c-min_sz)/2)
    return crop_square(im, start_r, start_c, min_sz)


def scale_to(x, ratio, targ):
    '''Calculate dimension of an image during scaling with aspect ratio'''
    return max(math.floor(x*ratio), targ)


def scale_min(im, targ, interpolation=cv2.INTER_AREA):
    """ Scales the image so that the smallest axis is of size targ.

    Arguments:
        im (array): image
        targ (int): target size
    """
    r,c,*_ = im.shape

    ratio = targ/min(r,c)

    sz = (scale_to(c, ratio, targ), scale_to(r, ratio, targ))

    # return im.resize(sz, resample=interpolation)
    return cv2.resize(im, sz, interpolation=interpolation)


# def rotate_img(img, deg, interpolation=Image.BICUBIC):
#     """ Rotate image by given angle """
#     return img.rotate(deg, resample=interpolation)

def rotate_img(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c//2,r//2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)


def pad(img, pad, mode=cv2.BORDER_REFLECT):
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, mode)


def mode_to_cv2(mode='constant'):
    if mode == 'constant':
        return cv2.BORDER_CONSTANT
    if mode == 'reflect':
        return cv2.BORDER_REFLECT

    raise ValueError(f"Invalid mode {mode}")


def lighting(im, b, c):
    ''' adjusts image's balance and contrast'''
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)
