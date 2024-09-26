import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from numba import cuda, float32, int32

#The entire code was run in Google Colab

directory = "/content/drive/MyDrive/NN_CUDA/"

multipliers = [np.load(directory + f"S_{i}.npy") for i in range(41, 45)] + \
              [np.load(directory + f"S_{i}.npy") for i in range(51, 55)] + \
              [np.load(directory + f"S_{i}.npy") for i in range(61, 65)] + \
              [np.load(directory + f"S_{i}.npy") for i in range(71, 75)] + \
              [np.load(directory + f"S_{i}.npy") for i in range(81, 85)]

mnist = tf.keras.datasets.mnist
(_, _), (test_images, test_labels) = mnist.load_data()

model_path = directory + "my_org_model_top4_quant.h5"
model = tf.keras.models.load_model(model_path)

model_weights = model.get_weights()

batch_size = 10
input_features = test_images[:batch_size]

# Define threads per block globally
threadsperblock = (16, 16)

@cuda.jit
def custom_matrix_multiplication_kernel(a, b, multiplier, result, t):
    row, col = cuda.grid(2)
    if row < result.shape[0] and col < result.shape[1]:
        tmp = 0
        for k in range(a.shape[1]):
            x = int(a[row, k]) + 128
            y = int(b[k, col]) + 128
            tmp += multiplier[x, y]
        result[row, col] = tmp

def custom_matrix_multiplication(a, b, t=1):
    a = np.array(a)
    b = np.array(b)
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)

    multiplier = multipliers[t - 1]

    blockspergrid_x = int(np.ceil(a.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(b.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    custom_matrix_multiplication_kernel[blockspergrid, threadsperblock](a, b, multiplier, result, t)

    return result

@cuda.jit
def custom_conv2d_kernel(a, b, multiplier, result, t):
    row, col = cuda.grid(2)
    if row < result.shape[0] and col < result.shape[1]:
        tmp = 0
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                x = int(a[row + i, col + j]) + 128
                y = int(b[i, j]) + 128
                tmp += multiplier[x, y]
        result[row, col] = tmp

def custom_conv2d(a, b, t=1):
    a = np.array(a)
    b = np.array(b)
    result_shape1 = np.abs(a.shape[0] - b.shape[0]) + 1
    result_shape2 = np.abs(a.shape[1] - b.shape[1]) + 1
    result = np.zeros((result_shape1, result_shape2), dtype=np.float32)

    multiplier = multipliers[t - 1]

    blockspergrid_x = int(np.ceil(result.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(result.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    custom_conv2d_kernel[blockspergrid, threadsperblock](a, b, multiplier, result, t)

    return result

def model_forward_pass(image_index, t=1):
    """
    Custom function to perform forward pass through the modified model.
    """
    # Perform a series of convolutions and activations
    conv1_input = np.floor(input_features[image_index] / 2)
    conv1_input = conv1_input.reshape(28, 28, 1)
    conv1_output = np.zeros([28, 28, 64])
    for i in range(64):
        for j in range(1):
            conv1_output[:, :, i] += custom_conv2d(np.array(conv1_input[:, :, j]), np.flip(model_weights[0][:, :, j, i]), t)
        conv1_output[:, :, i] += model_weights[1][i]
    relu1_output = np.maximum(0, conv1_output)
    relu1_output = np.round((relu1_output / np.max(relu1_output)) * 127)

    conv2_output = np.zeros([28, 28, 32])
    for i in range(32):
        for j in range(64):
            conv2_output[:, :, i] += custom_conv2d(np.array(relu1_output[:, :, j]), np.flip(model_weights[2][:, :, j, i]), t)
        conv2_output[:, :, i] += model_weights[3][i]
    relu2_output = np.maximum(0, conv2_output)
    relu2_output = np.round((relu2_output / np.max(relu2_output)) * 127)

    conv3_output = np.zeros([28, 28, 16])
    for i in range(16):
        for j in range(32):
            conv3_output[:, :, i] += custom_conv2d(np.array(relu2_output[:, :, j]), np.flip(model_weights[4][:, :, j, i]), t)
        conv3_output[:, :, i] += model_weights[5][i]
    relu3_output = np.maximum(0, conv3_output)
    relu3_output = np.round((relu3_output / np.max(relu3_output)) * 127)

    conv4_output = np.zeros([26, 26, 8])
    for i in range(8):
        for j in range(16):
            conv4_output[:, :, i] += custom_conv2d(np.array(relu3_output[:, :, j]), np.flip(model_weights[6][:, :, j, i]), t)
        conv4_output[:, :, i] += model_weights[7][i]
    relu4_output = np.maximum(0, conv4_output)
    relu4_output = np.round((relu4_output / np.max(relu4_output)) * 127)

    conv5_output = np.zeros([24, 24, 4])
    for i in range(4):
        for j in range(8):
            conv5_output[:, :, i] += custom_conv2d(np.array(relu4_output[:, :, j]), np.flip(model_weights[8][:, :, j, i]), t)
        conv5_output[:, :, i] += model_weights[9][i]
    relu5_output = np.maximum(0, conv5_output)
    relu5_output = np.round((relu5_output / np.max(relu5_output)) * 127)

    flatten_output = np.reshape(relu5_output, [1, 2304])
    fc1_output = custom_matrix_multiplication(flatten_output, model_weights[10], t) + model_weights[11]
    relu6_output = np.maximum(0, fc1_output) + 0.000001
    relu6_output = np.round((relu6_output / np.max(relu6_output)) * 127)

    fc2_output = custom_matrix_multiplication(relu6_output, model_weights[12], t) + model_weights[13]
    relu7_output = np.maximum(0, fc2_output) + 0.000001
    relu7_output = np.round((relu7_output / np.max(relu7_output)) * 127)

    fc3_output = custom_matrix_multiplication(relu7_output, model_weights[14], t) + model_weights[15] + 0.000001
    fc3_output = np.round((fc3_output / np.max(fc3_output)) * 127)

    return np.argmax(fc3_output)


# Loop through different multiplier types and calculate results

for multiplier_type in range(1, 21):
    
    start_time = time.time()

    results = []
    for image_index in range(10):
        results.append(model_forward_pass(image_index, multiplier_type))

    # Save results to file
    filename = f"Result_Approx_Multi_{multiplier_type}.npy"
    np.save(filename, results)

    # Print results and size once per multiplier type
    print(results)
    print(np.size(results))

    elapsed_time = time.time() - start_time
    print('Execution time:', elapsed_time, 'seconds')

    accuracy = np.sum(results == test_labels[:len(results)]) / len(results)
    print(f"Accuracy with multiplier {multiplier_type}: {accuracy}")
