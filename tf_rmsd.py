import tensorflow as tf
import numpy as np
import rmsd
from math import isclose
import pdb

def tf_kabsch(P, Q):
    # calculate covariance matrix
    C = tf.matmul(tf.transpose(P), Q)

    S, V, W = tf.svd(C, full_matrices=True, compute_uv=True)
    W = tf.linalg.adjoint(W)
    def cond1(S, V, m1, m2):
        # implement the following numpy ops in tensorflow
        # S[-1] = -S[-1]
        # V[:, -1] = -V[:, -1]
        return S * m1, V * m2
    def cond2(S, V):
        return S, V

    m1_np = np.ones((3,), np.float32)
    m1_np[-1] = -m1_np[-1]
    m1 = tf.constant(m1_np)

    m2_np = np.ones((3,3), np.float32)
    m2_np[:,-1] = -m2_np[:,-1]
    m2 = tf.constant(m2_np)

    d = tf.linalg.det(V) * tf.linalg.det(W)
    S, V = tf.cond(d < 0., lambda: cond1(S, V, m1, m2), lambda: cond2(S, V))
    # Rotation matrix U
    U = tf.matmul(V, W)
    return U

def tf_rmsd(V, W):
    N = tf.shape(V)[0]
    p1 = tf.reduce_sum((V - W) * (V - W), 1)
    return tf.sqrt(tf.reduce_sum(p1, 0) / tf.cast(N, tf.float32))

def tf_rmsd_masked(V, W, N):
    p1 = tf.reduce_sum((V - W) * (V - W), 1)
    return tf.sqrt(tf.reduce_sum(p1, 0) / tf.cast(N, tf.float32))


def tf_kabsch_rotate(P, Q):
    U = tf_kabsch(P, Q)
    # rotate matrix P
    return tf.matmul(P, U)

def tf_kabsch_rmsd(P, Q):
    P = tf_kabsch_rotate(P, Q)
    return tf_rmsd(P, Q)

def tf_kabsch_rmsd_masked(P, Q, mask, tol):
    N = tf.reduce_sum(mask)
    mask_mat = tf.diag(tf.reshape(mask, (-1,)))
    P_masked = tf.matmul(mask_mat, P) + tol
    Q_masked = tf.matmul(mask_mat, Q) + tol
    P_transformed = tf_kabsch_rotate(P_masked, Q_masked)
    return tf_rmsd_masked(P_transformed, Q_masked, N)

def tf_centroid(P):
    return tf.reduce_mean(P, axis=0, keepdims=True)

def tf_centroid_masked(P, mask, tol):
    N = tf.reduce_sum(mask)
    mask_mat = tf.diag(tf.reshape(mask, (-1,)))
    P_masked = tf.matmul(mask_mat, P) + tol
    return tf.reduce_sum(P_masked, axis=0, keepdims=True) / tf.cast(N, tf.float32)

if __name__ == "__main__":
    P = tf.placeholder(tf.float32, [None, 3])
    Q = tf.placeholder(tf.float32, [None, 3])
    P_cent = P - tf_centroid(P)
    Q_cent = Q - tf_centroid(Q)
    pq_rmsd = tf_kabsch_rmsd(P_cent, Q_cent)

    _, P_np = rmsd.get_coordinates_pdb('ci2_1.pdb')
    _, Q_np = rmsd.get_coordinates_pdb('ci2_2.pdb')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    qp_rmsd_tf = sess.run(pq_rmsd, feed_dict={P:Q_np, Q:P_np})
    qp_rmsd_np = rmsd.kabsch_rmsd(Q_np - rmsd.centroid(Q_np), P_np - rmsd.centroid(P_np))

    pq_rmsd_tf = sess.run(pq_rmsd, feed_dict={P:P_np, Q:Q_np})
    pq_rmsd_np = rmsd.kabsch_rmsd(P_np - rmsd.centroid(P_np), Q_np - rmsd.centroid(Q_np))


    print ("Kabsch RMSD(Q, P): Numpy implementation {}, TF implementation {}".format(qp_rmsd_np, qp_rmsd_tf))
    assert (isclose(qp_rmsd_tf, qp_rmsd_np, abs_tol=1e-5))

    print ("Kabsch RMSD(P, Q): Numpy implementation {}, TF implementation {}".format(pq_rmsd_np, pq_rmsd_tf))
    assert (isclose(pq_rmsd_tf, pq_rmsd_np, abs_tol=1e-5))
