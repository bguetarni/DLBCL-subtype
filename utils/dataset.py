import json
import numpy as np
import cv2
from skimage import filters, morphology


def load_contours_from_json(path_):

    def filter_(ctr):   # remove loops in object annotation
        
        if len(ctr) > 1:
            length_poly = [np.shape(i)[1] for i in ctr]
            return ctr[np.argmax(length_poly)]

        return ctr

    with open(path_, 'r') as f:
        fts = json.load(f)['features']

    has_name_property = ['name' in obj['properties'].keys() for obj in fts]
    if not all(has_name_property):
        raise RuntimeError('annotation file at {} has some annotations without name.')

    # gather objects id and coordinates
    objects = {}
    for obj in fts:
        ctr = filter_(obj['geometry']['coordinates'])
        ctr = np.array(ctr).reshape((-1, 2))
        objects.update({obj['properties']['name']: ctr})
    
    return objects


def scale_coordinates(wsi, p, source_level, target_level):

    if not isinstance(p, np.ndarray):
        p = np.asarray(p).squeeze()

    assert p.ndim < 3 and p.shape[-1] == 2, 'coordinates must be a single point or an array of 2D-cooridnates'

    # source level dimensions
    source_w, source_h = wsi.level_dimensions[source_level]
    
    # target level dimensions
    target_w, target_h = wsi.level_dimensions[target_level]
    
    # scale coordinates
    p = np.array(p)*(target_w/source_w, target_h/source_h)
    
    # round to int64
    return np.floor(p).astype('int64')


def get_tissue_mask(wsi, mask_level, black_threshold=90):
    """
    Args:
        slide : whole slide file (openslide.OpenSlide)
        mask_level : level from which to build the mask (int)

    Return np.ndarray of bool as mask
    """
    
    # get slide image into PIL.Image
    thumbnail = wsi.read_region((0, 0), mask_level, wsi.level_dimensions[mask_level])

    # convert to gray
    gray = np.array(thumbnail.convert('L'))
    
    # smooth image
    gray = filters.gaussian(gray, sigma=2.0, preserve_range=True)
    
    # computes Otsu threshold with blakish pixels filtered
    threshold = filters.threshold_otsu(gray[gray > black_threshold])
    
    # create tisue mask
    mask = gray < threshold
    
    # remove blackish pixels from mask
    mask[gray < black_threshold] = False

    # dilation to for removed nuclei (black) from last step
    mask = morphology.binary_dilation(mask)
    
    return mask


def get_tissue_mask_hsv(wsi, mask_level, black_threshold=100):
    """
    Args:
        slide : whole slide file (openslide.OpenSlide)
        mask_level : level from which to build the mask (int)

    Return np.ndarray of bool as mask
    """
    
    # get slide image into PIL.Image
    thumbnail = wsi.read_region((0, 0), mask_level, wsi.level_dimensions[mask_level])

    # convert to HSV
    hsv = np.array(thumbnail.convert('HSV'))
    H, S, V = np.moveaxis(hsv, -1, 0)

    # filter out black pixels
    V_mask = V > black_threshold

    S_filtered = S[V_mask]
    S_threshold = filters.threshold_otsu(S_filtered)
    S_mask = S > S_threshold

    H_filtered = H[V_mask]
    H_threshold = filters.threshold_otsu(H_filtered)
    H_mask = H > H_threshold

    mask = np.logical_and(np.logical_and(H_mask, S_mask), V_mask)
    mask = morphology.binary_dilation(mask)

    return mask


def create_mask_from_annotation(wsi, mask_level, annotation_path):

    # read contours annotations files
    contours = load_contours_from_json(annotation_path)
    
    # keep only contours points and scale to mask level
    contours = list(contours.values())
    contours = [scale_coordinates(wsi, ctr, 0, mask_level).astype('int32')for ctr in contours]

    # initiate mask with zeroes
    w, h = wsi.level_dimensions[mask_level]
    mask = np.zeros((h,w), dtype='uint8')

    # fill all annotations contours with 1 in mask
    mask = cv2.fillPoly(mask, contours, color=1).astype('bool')

    return mask


def create_object_mask(wsi, ctr, mask_level):

    # scale contour points to mask level
    pts = []
    for x, y in ctr:
        x, y = scale_coordinates(wsi, (x, y), 0, mask_level)
        pts.append((x, y))

    # initialize mask
    w, h = wsi.level_dimensions[mask_level]
    mask = np.zeros((h, w), dtype='uint8')
    
    # fill contours with 1s
    mask = cv2.fillPoly(mask, [np.array(pts)], color=1).astype('bool')
    
    return mask


def get_tile_mask(wsi, level, mask, mask_level, x, y, patch_size):
    
    # convert coordinates from slide level to mask level
    x_ul, y_ul = scale_coordinates(wsi, (x, y), level, mask_level)
    x_br, y_br = scale_coordinates(wsi, (x + patch_size, y + patch_size), level, mask_level)

    return mask[y_ul:y_br, x_ul:x_br]