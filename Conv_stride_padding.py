# Calculating Convolution using mask H of two channels H1, H2 for feature maps X1, X2 with different strides and padding

import numpy as np

def strided_len(x_len, H_len, stride):
    return np.ceil((x_len - H_len + 1) / stride).astype(int)


def H_dilated_len(H_len, dilation):
    return (H_len - 1) * (dilation - 1) + H_len


def dilate_H(H, dilation):
    H_rows, H_cols = H[0].shape
    H_dilated = np.zeros((H.shape[0], H_dilated_len(H_rows, dilation),
                          H_dilated_len(H_cols, dilation)))
    H_dilated[:, ::dilation, ::dilation] = H
    return H_dilated


def apply_mask(X, H, padding=0, stride=1, dilation=1):
    # x and H can have multiple channels in the 0th dimension
    if padding > 0:
        X = np.pad(X, pad_width=padding, mode='constant')[1:-1]

    if dilation > 1:
        H = dilate_H(H, dilation)

    H_rows, H_cols = H[0].shape
    x_rows, x_cols = X[0].shape

    fm_rows = strided_len(x_rows, H_rows, stride)
    fm_cols = strided_len(x_cols, H_cols, stride)

    feature_map = np.empty((fm_rows, fm_cols))
    for xf in range(fm_rows):
        for yf in range(fm_cols):
            xi, yi = xf * stride, yf * stride
            receptive_region = X[:, xi: xi + H_rows, yi: yi + H_cols]
            feature_map[xf, yf] = np.sum(H * receptive_region)
    return feature_map
  
if __name__ == '__main__':
  X1 = [[0.2, 1., 0.],
          [-1., 0., -0.1],
          [0.1, 0., 0.1]]

  X2 = [[1., 0.5, 0.2],
          [-1., -0.5, -0.2],
          [0.1, -0.1, 0.]]

  X = np.array([X1, X2])

  h1 = [[1., -0.1],
          [1., -0.1]]

  h2 = [[0.5, 0.5],
          [-0.5, -0.5]]

  H = np.array([h1, h2])
    # Calculate the output produced by mask H when using:
  print('padding=0 and stride=1')
  print(apply_mask(X, H))

  print('padding=1 and stride=1')
  print(apply_mask(X, H, padding=1))

  print('padding=1 and stride=2')
  print(apply_mask(X, H, padding=1, stride=2))

  print('padding=0 and stride=1 dilation=2')
  print(apply_mask(X, H, padding=0, stride=1, dilation=2))
