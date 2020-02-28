import argparse, sys
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngImageFile

PI = np.pi


def grayify(img: Image) -> np.ndarray:
    try:
        img = img.convert('L')
        return np.asanyarray(img).copy()
    except Exception as e:
        print(e)
        img = img.convert('LA')
        return np.asanyarray(img)[:][:][0].copy()


def gaussian_blur_(mtx: np.ndarray, sigma2: float, kernel_size: float = None) -> None:
    # Ensuring matrix dimension == 2
    assert mtx.ndim == 2

    # Ensuring array is writable
    assert mtx.flags['WRITEABLE']

    if not isinstance(kernel_size, int):
        kernel_size = int(3 * sigma2)
    if kernel_size % 2 == 0:
        kernel_size += 1
    half_ks = kernel_size // 2

    h, w = mtx.shape
    padded_mtx = np.pad(mtx, half_ks, 'reflect')

    # Init kernel
    double_sigma2 = 2 * sigma2
    x, y = np.mgrid[-half_ks:half_ks + 1, -half_ks:half_ks + 1]
    kernel = np.exp(-(x * x + y * y) / double_sigma2)
    del x, y, sigma2, double_sigma2, half_ks

    # Normalize kernel
    kernel /= kernel.sum()
    print('Kernel init succeeded!')

    # Convolve matrix
    for r in range(h):
        if r % 100 == 0:
            print(f'Blurred [{r / h:%}]!')
        for c in range(w):
            mtx[r][c] = kernel.ravel().dot(
                padded_mtx[r:r + kernel_size, c:c + kernel_size].ravel()
            )

    del h, w, kernel, kernel_size, padded_mtx, mtx


def gradient(mtx: np.ndarray) -> tuple:
    from math import atan2, hypot

    # Ensuring matrix dimension == 2
    assert mtx.ndim == 2

    sobel_x = np.asanyarray(((-1, +0, +1),
                             (-2, +0, +2),
                             (-1, +0, +1)), dtype=np.int16)
    sobel_y = np.asanyarray(((-1, -2, -1),
                             (+0, +0, +0),
                             (+1, +2, +1)), dtype=np.int16)

    h, w = mtx.shape
    e_s = np.empty((h, w))
    e_0 = np.empty((h, w))
    sobel_kernel_size = 3
    padded_mtx = np.pad(mtx, sobel_kernel_size // 2, 'reflect')

    for r in range(h):
        if r % 100 == 0:
            print(f'Gradiented [{r / h:%}]!')
        for c in range(w):
            window = padded_mtx[r:r + sobel_kernel_size, c:c + sobel_kernel_size].ravel()
            j_x = sobel_x.ravel().dot(window)
            j_y = sobel_y.ravel().dot(window)
            e_s[r][c] = hypot(j_x, j_y)
            e_0[r][c] = atan2(j_y, j_x)

    e_s = e_s * 255 / e_s.max() + .5  # rounding
    e_s = e_s.astype(np.uint8, copy=False)
    del h, w, sobel_kernel_size, padded_mtx, mtx
    return e_s, e_0


# noinspection PyBroadException
def non_max_suppress(mtx: np.ndarray, e_s: np.ndarray, e_0: np.ndarray) -> np.ndarray:
    # Ensuring matrix dimension == 2
    assert mtx.ndim == 2

    i_n: np.ndarray = e_s.copy()

    h, w = mtx.shape
    e_0 *= 8 / PI  # range [-8, 8]

    for r in range(h):
        if r % 100 == 0:  # progress indicator
            print(f'Suppressed [{r / h:%}]!')
        for c in range(w):
            rad = e_0[r][c]
            if rad < 0:
                rad += 8  # range [0, 8]

            if rad < 1 or rad > 7:
                offset = (+0, +1)
            elif rad < 3:
                offset = (+1, +1)
            elif rad < 5:
                offset = (+1, +0)
            else:  # rad < 7
                offset = (-1, +1)

            try:
                if e_s[r][c] < e_s[r + offset[0]][c + offset[1]]:
                    i_n[r][c] = 0
                    continue
            except:
                ...
            try:
                if e_s[r][c] < e_s[r - offset[0]][c - offset[1]]:
                    i_n[r][c] = 0
                    continue
            except:
                ...

    del mtx, h, w, r, c, rad, offset
    return i_n


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        parser = argparse.ArgumentParser()
        parser.add_argument('input_file')
        parser.add_argument('sigma2', type=float)
        parser.add_argument('output_file')
        args = parser.parse_args()

        input_file = args.input_file
        _sigma2 = args.sigma2
        output_file = args.output_file
        del parser, args
    else:
        input_file = 'input.png'
        _sigma2 = 2
        output_file = 'output.png'

    try:
        import cv2

        img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.Canny(img, 64, 256 - 64)
        cv2.imwrite(f'{output_file}.cv2.png', img)
    except:
        ...

    input_img: PngImageFile = Image.open(input_file)

    img_arr = grayify(input_img)

    blur_img_arr = img_arr
    del img_arr
    gaussian_blur_(blur_img_arr, _sigma2)
    Image.fromarray(blur_img_arr).save(f'{output_file}._gauss.png')

    _e_s, _e_0 = gradient(blur_img_arr)
    Image.fromarray(_e_s).save(f'{output_file}._without-non-max.png')

    _i_n = non_max_suppress(blur_img_arr, _e_s, _e_0)

    Image.fromarray(_i_n).save(output_file)

    print('Done!')
