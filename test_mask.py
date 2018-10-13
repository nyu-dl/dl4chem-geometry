import numpy as np
import tensorflow as tf
from tf_rmsd import tf_kabsch_rmsd_masked, tf_kabsch_rmsd, tf_kabsch_rotate, tf_centroid, tf_centroid_masked

def mask_svd(frames, masks):
    batch_size = frames.shape[0]

    padding = tf.constant([[0, 0], [0, 0], [0, 1]])
    padded_frames = tf.pad(frames, padding, 'constant', constant_values=1)

    mask_matrices = []
    for i in range(batch_size):
        mask_matrix = tf.diag(tf.reshape(masks[i], [-1]))
        mask_matrices.append(mask_matrix)
    mask_tensor = tf.stack(mask_matrices)
    masked_frames = tf.matmul(mask_tensor, padded_frames)
    svd_in = masked_frames[0]
    return tf.svd(svd_in), svd_in

if __name__ == '__main__':
    test_type = 2
    if test_type == 1:
        frame = tf.placeholder(tf.float32, [1, 9, 3])
        mask = tf.placeholder(tf.float32, [1, 9, 1])
        output = mask_svd(frame, mask)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        unmasked_points = 7
        fval = np.random.rand(1, 9, 3)
        tval = np.random.rand(1, 9, 3)
        mval = np.zeros((1, 9, 1))
        mval[0, :unmasked_points, :] = 1

        #transformed_frame_val = sess.run(output, feed_dict={frame: fval, target: tval, mask: mval})
        out, svd_in_val = sess.run(output, feed_dict={frame: fval, mask: mval})
        sval, uval, vval = out

        tframes = tf.placeholder(tf.float32, [7, 3])

        padding = tf.constant([[0, 0], [0, 1]])
        padded_tframes = tf.pad(tframes, padding, 'constant', constant_values=1)

        toutput = tf.svd(padded_tframes)
        tout, tsvd_in_val = sess.run((toutput, padded_tframes), feed_dict={tframes: fval[0, :unmasked_points, :],})
        tsval, tuval, tvval = tout
        #assert(np.all(np.isclose(transformed_frame_val[0, :unmasked_points, :], transformed_tframe_val, atol=1e-5)))
        #assert (np.all(np.isclose(transformed_frame_val[0, :, :unmasked_points], transformed_tframe_val, atol=1e-5)))

        assert(np.all(np.isclose(sval, tsval, atol=1e-5)))
        assert (np.all(np.isclose(vval, tvval, atol=1e-5)))
        assert (np.all(np.isclose(uval[:unmasked_points], tuval, atol=1e-5)))
        print('end')
    elif test_type == 2:
        frame = tf.placeholder(tf.float32, [9, 3])
        target = tf.placeholder(tf.float32, [9, 3])
        mask = tf.placeholder(tf.float32, [9, 1])

        output = tf_kabsch_rmsd_masked(target - tf_centroid_masked(target, mask), frame - tf_centroid_masked(frame, mask), mask)
        #output = tf_kabsch_rotate(target, frame)
        unmasked_points = 7
        fval = np.random.rand(9, 3)
        tval = np.random.rand(9, 3)
        tval[unmasked_points:] = 0
        mval = np.ones((9, 1))
        mval[unmasked_points:] = 0
        sess = tf.Session()
        outval = sess.run((output), feed_dict={frame: fval, target: tval, mask:mval})

        tframe = tf.placeholder(tf.float32, [7, 3])
        ttarget = tf.placeholder(tf.float32, [7, 3])
        toutput = tf_kabsch_rmsd(ttarget - tf_centroid(ttarget), tframe - tf_centroid(tframe))
        #toutput = tf_kabsch_rotate(ttarget, tframe)
        toutval = sess.run(toutput, feed_dict={tframe: fval[:unmasked_points], ttarget: tval[:unmasked_points]})

        assert outval == toutval