import tensorflow as tf

from PIL import Image

original_image_list = ["imgs/1.jpg", "imgs/2.png", "imgs/3.png", "imgs/4.jpg"]

# Make a queue of the file names including all the images specified
filename_queue = tf.train.string_input_producer(original_image_list)

# Read an entire image file
image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_list = []
    for i in range(len(original_image_list)):
        # Read a whole file from the queue, the first returned value in the tuple is the
        # filename which we are ignoring.
        _, image_file = image_reader.read(filename_queue)

        # Decode the image as a JEPG file, this will turn it into a Tensor which we can
        # then use in training.
        image = tf.image.decode_jpeg(image_file)

        # Get a tensor of resized images.
        image = tf.image.resize_images(image, [224, 224])

        image.set_shape((224, 224, 3))

        # Get an image tensor and print its value.
        image_array = sess.run(image)
        print(image_array.shape)

        Image.fromarray(image_array.astype('uint8'), 'RGB').show()

        # The expand_dims adds a new dimention
        image_list.append(tf.expand_dims(image_array, 0))

    # Finish off the filename queue coodinators
    coord.request_stop()
    coord.join(threads)

    index = 0

    # Writer image summary
    summary_writer = tf.summary.FileWriter("graphs", graph=sess.graph)

    for image_tensor in image_list:
        summary_str = sess.run(tf.summary.image("image-" + str(input), image_tensor))
        summary_writer.add_summary(summary_str)
        index += 1

    summary_writer.close()
