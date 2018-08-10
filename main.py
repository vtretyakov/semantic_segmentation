#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
import numpy as np
from moviepy.editor import VideoFileClip

clip = VideoFileClip("challenge_video.mp4")

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # load the model and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # extract the layers from the model
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return w1, keep_prob, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # do 1x1 convolutions on layers 7, 4, and 3 with a regularizer for the weights
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                       kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
                                   
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                       kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
                                       
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                       kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
                                   
    # upsample starting with layer 7
    output = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, 4, 2, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
                                        
    # add skip connection from layer 4 and upsample
    output = tf.add(output, conv_1x1_layer4)
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
                                        
    # add skip connection from layer 3 and upsample
    output = tf.add(output, conv_1x1_layer3)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    # the output tensor is 4D so we have to reshape it to 2D
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    # loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    
    #training
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    
    # initialize our global variables
    sess.run(tf.global_variables_initializer())
    
    # start timer
    start_time = time.time()
    print("Training...")
    print()
    
    #define a log array
    log = []

    # go through all epochs
    for epoch in range(epochs):
        #print("Epoch {}".format(epoch + 1), "/ {} ..".format(epochs))

        # go through all batches
        batch = 1
        losses = []
        for image, label in get_batches_fn(batch_size):
            print("Epoch {}".format(epoch + 1), "/ {} ".format(epochs), "Batch {} ..".format(batch))
            batch += 1
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0001})
            print("Loss: = {:.3f}".format(loss))
            losses.append(loss)
        # calculate average loss
        loss_sum = sum(losses)
        avg_loss = loss_sum / len(losses)

        # stop timer and print training results
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        elapsed_time = int(round(elapsed_time))
        print()
        print("Epoch {}".format(epoch+1), "Results:")
        print("Epoch: {}/{} | Total Time: {} mins | Avg Loss: {:.2f}".format(epoch+1, epochs, elapsed_time, avg_loss))
        print()
        log.append((epoch+1, avg_loss))
    np.savetxt('learning_log.csv', log, fmt='%.3f', delimiter=',')
tests.test_train_nn(train_nn)


def run(video_image):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    
    epochs = 20 #25
    batch_size = 2 #16

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.gen_video_output(sess, image_shape, logits, keep_prob, video_image, image_shape)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    new_clip = clip.fl_image( run )
    new_clip.write_videofile("challenge_video_processed.mp4", audio=False)
