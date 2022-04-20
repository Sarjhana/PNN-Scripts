
# Returns output of the height, width and output size based on input shape,
# mask shape, number of masks, stride and padding. When defining shape for
# input or mask specify it as (height,width, n_channels)
def calc_output_dim(input_shape, mask_shape, n_masks, stride, padding):
    output_h = int(calc_dim(input_shape[0], mask_shape[0], padding, stride))
    output_w = int(calc_dim(input_shape[1], mask_shape[1], padding, stride))

    return output_h, output_w, output_h * output_w * n_masks


# Returns output size for specific dimension, like height or width in case of
# 2D arrays, based on input dimension, mask dimension, padding and stride
def calc_dim(input_dim, mask_dim, padding, stride):
    return 1 + ((input_dim - mask_dim + 2 * padding) / stride)
  

if __name__ == '__main__':

    # Tutorial Q9 The input to a convolutional layer of a CNN consists of 6 feature
    # maps each of which has a height of 11 and width of 15 (i.e., input is
    # 11 × 15 × 6). What size will the output produced by a single mask with
    # 6 channels and a width of 3 and a height of 3 (i.e., 3×3×6) when using
    # a stride of 2 and padding of 0.
    
    # define shape using template (height,width,n_channels)
    
    input_shape = (11, 15, 6)
    mask_shape = (3, 3, 6)
    n_masks = 1
    stride = 2
    padding = 0
    print('calculate output dimension')
    out_h, out_w, out_size = calc_output_dim(input_shape, mask_shape, n_masks,
                                             stride, padding)
    print('output shape is: ' + str(out_h) + 'x' + str(out_w) + 'x' +
          str(n_masks) + '=' + str(out_size))
