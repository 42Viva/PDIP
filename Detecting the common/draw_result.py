import numpy as np
from PIL import Image


def draw_result(src_img, sig_score_img, region_size, scale):
    ma = np.max(sig_score_img)
    mi = np.min(sig_score_img)

    norm_sig_score_img = (sig_score_img - mi) / (ma - mi)
    norm_sig_score_img = norm_sig_score_img * 255
    norm_sig_score_img = np.array(Image.fromarray(norm_sig_score_img.astype('uint8')).resize(
        (int(norm_sig_score_img.shape[1] * scale), int(norm_sig_score_img.shape[0] * scale))))

    Image.fromarray(norm_sig_score_img.astype('uint8')).show()

    x, y = np.where(sig_score_img == ma)
    rect_img = np.zeros_like(sig_score_img)
    rect_img[x[0] - int(region_size[0] / 2):x[0] + int(region_size[0] / 2) + 1,
    y[0] - int(region_size[1] / 2):y[0] + int(region_size[1] / 2) + 1] = 128
    rect_img = np.array(Image.fromarray(rect_img.astype('uint8')).resize((src_img.shape[1], src_img.shape[0])))
    src_img[:, :, 0] = src_img[:, :, 0] + rect_img[:, :, 0]

    Image.fromarray(src_img.astype('uint8')).show()