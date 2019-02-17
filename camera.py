from __future__ import division
import tensorflow as tf
import numpy as np
import cv2

import model

def main():
    checkpoint_dir = './checkpoint/'

    ratio = 100

    cap = cv2.VideoCapture(0)

    # show origin view
    while True:
        _, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    with tf.Session(graph=tf.Graph()) as sess:

        in_image = tf.placeholder(tf.float32, [None, None, None, 3])
        out_image = model.unet(in_image)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt:
            print('loaded ' + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # show processed view
        while True:
            _, frame = cap.read()
            frame = frame.astype(np.uint8)
            input_full = np.expand_dims(np.float32(frame/255.0), axis = 0) * ratio
            input_full = np.minimum(input_full, 1.0)
            output = sess.run(out_image, feed_dict={in_image: input_full})
            output = np.minimum(np.maximum(output, 0), 1)
            output = output[0, :, :, :]
            cv2.imshow('frame', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__=='__main__':
    main()