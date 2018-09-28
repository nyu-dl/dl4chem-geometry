#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>

using namespace tensorflow;

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("PairwiseMSD")
        .Input("confs1: float32")
        .Input("confs2: float32")
        .Output("pairwise_msd: float32")
        .Output("rotation_mats: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext * c) {
            ShapeHandle sh1, sh2;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &sh1));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &sh2));
            auto msd_shape = c->Matrix(c->Dim(sh1, 0), c->Dim(sh2, 0));
            c->set_output(0, msd_shape);

            auto rot_shape = c->MakeShape({c->Dim(sh1, 0), c->Dim(sh2, 0), 3, 3});
            c->set_output(1, rot_shape);
            return Status::OK();
        });


#include <tensorflow/core/framework/op_kernel.h>
#include "theobald_rmsd.h"
#include "center.h"

class PairwiseMSDOp : public OpKernel {
public:
    explicit PairwiseMSDOp(OpKernelConstruction * context) : OpKernel(context) {

    }

    void Compute(OpKernelContext * context) override {
        // Grab the input tensor
        const Tensor & confs1 = context->input(0);
        const Tensor & confs2 = context->input(1);

        const int n_atoms = static_cast<int>(confs1.shape().dim_size(1));
        const int n_atoms2 = static_cast<int>(confs2.shape().dim_size(1));
        OP_REQUIRES(context, n_atoms == n_atoms2,
                    errors::InvalidArgument("Number of atoms must be the same: ", n_atoms, " and ", n_atoms2));

        int64 n_dim = confs1.shape().dim_size(2);
        OP_REQUIRES(context, n_dim == 3,
                    errors::InvalidArgument("Number of dimensions (1) must be three: ", n_dim));
        n_dim = confs2.shape().dim_size(2);
        OP_REQUIRES(context, n_dim == 3,
                    errors::InvalidArgument("Number of dimensions (2) must be three: ", n_dim));

        const int64 N1 = confs1.shape().dim_size(0);
        const int64 N2 = confs2.shape().dim_size(0);
        TensorShape output_shape{N1, N2};
        TensorShape rot_output_shape{N1, N2, 3, 3};
        TensorShape g1_shape{N1};
        TensorShape g2_shape{N2};

        auto flatconfs1 = confs1.flat<float>();
        auto flatconfs2 = confs2.flat<float>();

        Tensor * output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        auto output = output_tensor->matrix<float>();

        Tensor * rot_output_tensor = nullptr;
        context->allocate_output(1, rot_output_shape, &rot_output_tensor);
        auto rot_output = rot_output_tensor->flat<float>();

        Tensor confcopy1(DT_FLOAT, TensorShape({}));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, confs1.shape(), &confcopy1));
        auto confcopyflat1 = confcopy1.flat<float>();
        Tensor confcopy2(DT_FLOAT, TensorShape({}));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, confs2.shape(), &confcopy2));
        auto confcopyflat2 = confcopy2.flat<float>();

        Tensor g1(DT_FLOAT, TensorShape({}));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, g1_shape, &g1));
        auto g1v = g1.vec<float>();
        Tensor g2(DT_FLOAT, TensorShape({}));
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, g2_shape, &g2));
        auto g2v = g2.vec<float>();

        // TODO: Use mutable input
        for (int conf_i = 0; conf_i < N1; conf_i++) {
            for (int atom_i = 0; atom_i < n_atoms; atom_i++) {
                for (int dim_i = 0; dim_i < 3; dim_i++) {
                    confcopyflat1(conf_i * n_atoms * 3 + atom_i * 3 + dim_i) = \
                        flatconfs1(conf_i * n_atoms * 3 + atom_i * 3 + dim_i);
                }
            }
        }
        for (int conf_i = 0; conf_i < N2; conf_i++) {
            for (int atom_i = 0; atom_i < n_atoms; atom_i++) {
                for (int dim_i = 0; dim_i < 3; dim_i++) {
                    confcopyflat2(conf_i * n_atoms * 3 + atom_i * 3 + dim_i) = \
                        flatconfs2(conf_i * n_atoms * 3 + atom_i * 3 + dim_i);
                }
            }
        }


        inplace_center_and_trace_atom_major(confcopyflat1.data(), g1v.data(), static_cast<int>(N1), n_atoms);
        inplace_center_and_trace_atom_major(confcopyflat2.data(), g2v.data(), static_cast<int>(N2), n_atoms);

        int i = 0;
        int j = 0;
#pragma omp parallel for collapse(2)
        for (i = 0; i < N1; i++) {
            for (j = 0; j < N2; j++) {
                float msd = msd_atom_major(n_atoms, n_atoms,
                                           &confcopyflat1.data()[i * n_atoms * 3],
                                           &confcopyflat2.data()[j * n_atoms * 3],
                                           g1v(i), g2v(j),
                                           1, &rot_output.data()[i * N2 * 9 + j * 9]);
                output(i, j) = msd;
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("PairwiseMSD").Device(DEVICE_CPU), PairwiseMSDOp);
