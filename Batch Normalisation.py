
# Returns samples after batch normalization.
# Parameters: samples - collections of samples in which every sample needs to
# have the same shape, beta, gamma, epsilon are parameters of the batch
# normalization
def batch_normalization(samples, beta, gamma, epsilon):
    output = []
    shape = samples[0].shape

    means = np.zeros(shape)
    variances = np.zeros(shape)
    for y in range(shape[1]):
        for x in range(shape[0]):
            values_at_x_y = np.array([sample[y][x] for sample in samples])
            mean = np.mean(values_at_x_y)
            variance = np.var(values_at_x_y)

            means[y][x] = mean
            variances[y][x] = variance

    for sample in samples:
        b_normalized_sample = np.zeros(shape)
        for y in range(shape[1]):
            for x in range(shape[0]):
                value = sample[y][x]
                b_normalized_sample[y][x] = calc_b_norm(beta, epsilon, gamma,
                                                        means[y][x], value,
                                                        variances[y][x])
        output.append(b_normalized_sample)

    return output


# Returns batch normalization for a single value
def calc_b_norm(beta, epsilon, gamma, mean, value, variance):
    return beta + gamma * ((value - mean) / np.sqrt(variance + epsilon))

  
if __name__ == '__main__':
  X1 = np.array([[1, 0.5, 0.2],
                   [-1, -0.5, -0.2],
                   [0.1, -0.1, 0]])

  X2 = np.array([[1, -1, 0.1],
                   [0.5, -0.5, -0.1],
                   [0.2, -0.2, 0]])

  X3 = np.array([[0.5, -0.5, -0.1],
                   [0, -0.4, 0],
                   [0.5, 0.5, 0.2]])

  X4 = np.array([[0.2, 1, -0.2],
                   [-1, -0.6, -0.1],
                   [0.1, 0, 0.1]])

  for output in batch_normalization([X1, X2, X3, X4], 0, 1, 0.1):
      print(str(output) + '\n')
