import numbers
import cv2
import numpy as np
import PIL
import torch


def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip, np.ndarray):
        cropped = clip[:, min_h:min_h + h, min_w:min_w + w, :]
    
    elif isinstance(clip, torch.Tensor):
        cropped = clip[:, min_h:min_h + h, min_w:min_w + w, :]
    
    elif isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def resize_clip(clip, size, interpolation='bilinear'):
    # if isinstance(clip[0], np.ndarray):
    #     if isinstance(size, numbers.Number):
    #         im_h, im_w, im_c = clip[0].shape
    #         # Min spatial dim already matches minimal size
    #         if (im_w <= im_h and im_w == size) or (im_h <= im_w
    #                                                and im_h == size):
    #             return clip
    #         new_h, new_w = get_resize_sizes(im_h, im_w, size)
    #         size = (new_w, new_h)
    #     else:
    #         size = size[0], size[1]
    #     if interpolation == 'bilinear':
    #         np_inter = cv2.INTER_LINEAR
    #     else:
    #         np_inter = cv2.INTER_NEAREST
    #     scaled = [
    #         cv2.resize(img, size, interpolation=np_inter).astype(np.uint8) for img in clip
    #     ]
    if isinstance(clip, np.ndarray):
        b, im_h, im_w, im_c = clip.shape
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
    
    if isinstance(clip, torch.Tensor):
        clip = clip.permute(0, 3, 1, 2).float()

        if isinstance(size, numbers.Number):
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[0], size[1]
            
        clip = torch.nn.functional.interpolate(clip, size = (size[1], size[0]), mode=interpolation)
        clip = clip.permute(0, 2, 3, 1).long()
        # clip = clip.numpy().astype(np.uint8)
        scaled = clip
        
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the video clip
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, H, W, C)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized clip
    Returns:
        clip (torch.tensor): Resized and cropped clip. Size is (T, H, W, C)
    """
    if not _is_tensor_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    clip = crop_clip(clip, i, j, h, w)
    clip = resize_clip(clip, size, interpolation_mode)
    return clip


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def normalize(clip, mean, std, inplace=True):
    if not _is_tensor_clip(clip):
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip