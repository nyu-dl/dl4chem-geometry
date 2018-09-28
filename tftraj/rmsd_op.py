import os
import pkg_resources
import tensorflow as tf
import warnings
from tensorflow.python.framework import ops


def load(debug=False):
    so = pkg_resources.resource_filename(__name__, 'rmsd/librmsd.Release.so')
    if debug or not os.path.isfile(so):
        so = pkg_resources.resource_filename(__name__, 'rmsd/librmsd.Debug.so')
        if not os.path.isfile(so):
            raise FileNotFoundError("Could not find the RMSD op shared library. "
                                    "Make sure you compile it!")
        warnings.warn("Using debug build of RMSD Op. This will be slow!")
    mod = tf.load_op_library(so)
    return mod


def drot_dcentered(confs1, confs2, rots, grad):
    N1 = tf.shape(confs1)[0]
    N2 = tf.shape(confs2)[0]
    n_atom = float(int(confs1.get_shape()[1]))

    expand_confs1 = tf.expand_dims(confs1, axis=1)
    big_confs1 = tf.tile(expand_confs1, [1, N2, 1, 1])
    expand_confs2 = tf.expand_dims(confs2, axis=0)
    big_confs2 = tf.tile(expand_confs2, [N1, 1, 1, 1])

    dxy = expand_confs1 - tf.matmul(big_confs2, rots, transpose_b=True)
    dxy = 2 * dxy / n_atom

    dyx = expand_confs2 - tf.matmul(big_confs1, rots, transpose_b=False)
    dyx = 2 * dyx / n_atom

    grad = tf.expand_dims(tf.expand_dims(grad, axis=-1), axis=-1)
    dr_dc1 = tf.reduce_sum(grad * dxy, axis=1)
    dr_dc2 = tf.reduce_sum(grad * dyx, axis=0)
    return dr_dc1, dr_dc2


def dcenter_dx(confs1, confs2, dr_dc1, dr_dc2):
    # Gradients for centering at origin
    centered1 = confs1 - tf.reduce_mean(confs1, axis=1, keep_dims=True)
    centered2 = confs2 - tf.reduce_mean(confs2, axis=1, keep_dims=True)
    center_grad1 = tf.gradients(centered1, [confs1], grad_ys=dr_dc1)[0]
    center_grad2 = tf.gradients(centered2, [confs2], grad_ys=dr_dc2)[0]
    return center_grad1, center_grad2


@ops.RegisterGradient("PairwiseMSD")
def _pairwise_msd_grad(op, grad, rot_grad):
    # TODO: Throw error if rot_grad is non-zero?
    confs1, confs2 = op.inputs
    rots = op.outputs[1]
    dr_dc1, dr_dc2 = drot_dcentered(confs1, confs2, rots, grad)
    dc_dx1, dc_dx2 = dcenter_dx(confs1, confs2, dr_dc1, dr_dc2)
    return dc_dx1, dc_dx2
