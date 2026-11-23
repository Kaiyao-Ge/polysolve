#include "ReducedMatrix.hpp"
#include "../ParallelKernels.hpp"

#include <Eigen/Dense>

#include <algorithm>

namespace polysolve::nonlinear
{
    void ReducedMatrix::build(const Mat &Q, const Mat &G, const NonsmoothEval &c, const int k, Mat &A) const
    {
        if (c.local)
        {
            // S += L^T H L for the blocks [b0, b1) in order (L: the rows of Q of the block)
            const auto add_blocks = [&](Mat &S, const Eigen::Index b0, const Eigen::Index b1) {
                Eigen::Matrix<double, HessianBlock::max_size, Eigen::Dynamic> LQ(HessianBlock::max_size, k);
                for (Eigen::Index ib = b0; ib < b1; ++ib)
                {
                    const auto &b = c.blocks[ib];
                    LQ.setZero();
                    bool any = false;
                    for (int i = 0; i < b.size; ++i)
                        if (b.indices[i] >= 0)
                        {
                            LQ.row(i) = Q.block(b.indices[i], 0, 1, k);
                            any = true;
                        }
                    if (any)
                        S.noalias() += LQ.topRows(b.size).transpose() * (b.H.topLeftCorner(b.size, b.size) * LQ.topRows(b.size));
                }
            };
            // One thread (or fewer than 128 blocks): A = G, then the blocks in order, as always. Several
            // threads: the blocks in contiguous chunks whose number depends only on the number of blocks, one sum
            // per chunk (the first starts from G) added in chunk order, with MKL sequential inside a chunk (its
            // own threading only adds dispatch cost for these small products). The result then does not depend
            // on the number of threads, but it rounds differently from the one-thread sum.
            const Eigen::Index nb = Eigen::Index(c.blocks.size());
            const int nc = parallel::max_threads() > 1 ? int(std::clamp<Eigen::Index>(nb / 64, 1, 64)) : 1;
            if (nc == 1)
            {
                A = G;
                add_blocks(A, 0, nb);
            }
            else
            {
                std::vector<Mat> acc(nc);
                parallel::for_chunks(nb, nc, [&](const int ci, const Eigen::Index b0, const Eigen::Index b1) {
#ifdef EIGEN_USE_MKL
                    const int mkl_threads = mkl_set_num_threads_local(1);
#endif
                    acc[ci] = ci == 0 ? G : Mat(Mat::Zero(k, k));
                    add_blocks(acc[ci], b0, b1);
#ifdef EIGEN_USE_MKL
                    mkl_set_num_threads_local(mkl_threads);
#endif
                });
                A = std::move(acc[0]);
                for (int i = 1; i < nc; ++i)
                    A += acc[i];
            }
        }
        else
        {
            const Mat KQ = c.K * Q.leftCols(k);
            A = G + Q.leftCols(k).transpose() * KQ;
        }
    }
} // namespace polysolve::nonlinear
