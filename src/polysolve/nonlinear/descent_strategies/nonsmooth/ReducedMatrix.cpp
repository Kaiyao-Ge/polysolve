#include "ReducedMatrix.hpp"
#include "../ParallelKernels.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>

namespace polysolve::nonlinear
{
    // The structure of the merged form: the blocks of the first point of an outer iteration (distinct index lists).
    struct ReducedMatrix::Structure
    {
        struct Blk
        {
            int m = 0;
            std::array<int, HessianBlock::max_size> ord{};
            long off = 0; ///< position of its m x m table in pos
        };
        std::vector<int> R, cid;       ///< reduced dof of each compact row; reduced dof -> compact row (-1)
        std::vector<int> outer, inner; ///< CSR of K_R (row-major), fixed within the outer iteration
        std::vector<double> vals;
        std::vector<Blk> blk;          ///< the distinct blocks
        std::vector<int> pos;          ///< per block m x m entries: position in vals (-1: an entry of a fixed dof)
        std::unordered_multimap<uint64_t, int> by_hash;
        Mat QR, W;
    };

    namespace
    {
        uint64_t hash_order(const int *idx, const int m)
        {
            uint64_t h = 1469598103934665603ull;
            for (int i = 0; i < m; ++i)
                h = (h ^ uint64_t(uint32_t(idx[i]))) * 1099511628211ull;
            return h;
        }
    } // namespace

    void ReducedMatrix::begin_outer()
    {
        first_ = true;
        merged_ = false;
        t_.reset();
    }

    long ReducedMatrix::structure_nnz() const { return t_ ? long(t_->inner.size()) : 0; }
    std::vector<int> ReducedMatrix::structure_rows() const { return t_ ? t_->R : std::vector<int>(); }

    void ReducedMatrix::add_blocks(const Mat &Q, const NonsmoothEval &c, const int k, Mat &S, const Eigen::Index b0, const Eigen::Index b1) const
    {
        // S += L^T H L for the blocks [b0, b1) in order (L: the rows of Q of the block)
        Eigen::Matrix<double, HessianBlock::max_size, Eigen::Dynamic> LQ(HessianBlock::max_size, k);
        for (Eigen::Index ib = b0; ib < b1; ++ib)
        {
            const auto &b = c.blocks[ib];
            LQ.setZero();
            bool any = false;
            for (int i = 0; i < b.size; ++i)
                if (b.indices[i] >= 0)
                {
                    gather(Q, b.indices[i], k, LQ.row(i));
                    any = true;
                }
            if (any)
                S.noalias() += LQ.topRows(b.size).transpose() * (b.H.topLeftCorner(b.size, b.size) * LQ.topRows(b.size));
        }
    }

    void ReducedMatrix::make_structure(const NonsmoothEval &c, const Eigen::Index n)
    {
        auto T = std::make_shared<Structure>();
        T->cid.assign(size_t(n), -1);
        for (const HessianBlock &hb : c.blocks)
            for (int i = 0; i < hb.size; ++i)
                if (const int r = hb.indices[i]; r >= 0 && T->cid[size_t(r)] < 0)
                {
                    T->cid[size_t(r)] = int(T->R.size());
                    T->R.push_back(r);
                }
        const int nR = int(T->R.size());
        std::vector<std::vector<int>> cols(static_cast<size_t>(nR));
        for (const HessianBlock &hb : c.blocks)
            for (int i = 0; i < hb.size; ++i)
                if (hb.indices[i] >= 0)
                    for (int j = 0; j < hb.size; ++j)
                        if (hb.indices[j] >= 0)
                            cols[size_t(T->cid[size_t(hb.indices[i])])].push_back(T->cid[size_t(hb.indices[j])]);
        T->outer.assign(size_t(nR) + 1, 0);
        for (int r = 0; r < nR; ++r)
        {
            auto &v = cols[size_t(r)];
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            T->outer[size_t(r) + 1] = T->outer[size_t(r)] + int(v.size());
            T->inner.insert(T->inner.end(), v.begin(), v.end());
        }
        T->vals.assign(T->inner.size(), 0.0);
        for (const HessianBlock &hb : c.blocks)
        {
            Structure::Blk B;
            B.m = hb.size;
            B.ord.fill(-1);
            for (int i = 0; i < hb.size; ++i)
                B.ord[size_t(i)] = hb.indices[i];
            const uint64_t h = hash_order(hb.indices.data(), hb.size);
            bool dup = false;
            const auto rg = T->by_hash.equal_range(h);
            for (auto it = rg.first; it != rg.second && !dup; ++it)
                dup = T->blk[size_t(it->second)].m == B.m && T->blk[size_t(it->second)].ord == B.ord;
            if (dup)
                continue;
            B.off = long(T->pos.size());
            for (int i = 0; i < hb.size; ++i)
                for (int j = 0; j < hb.size; ++j)
                {
                    if (hb.indices[i] < 0 || hb.indices[j] < 0)
                    {
                        T->pos.push_back(-1);
                        continue;
                    }
                    const int row = T->cid[size_t(hb.indices[i])], col = T->cid[size_t(hb.indices[j])];
                    const auto b0 = T->inner.begin() + T->outer[size_t(row)], b1 = T->inner.begin() + T->outer[size_t(row) + 1];
                    T->pos.push_back(int(std::lower_bound(b0, b1, col) - T->inner.begin()));
                }
            T->by_hash.emplace(h, int(T->blk.size()));
            T->blk.push_back(B);
        }
        t_ = T;
    }

    void ReducedMatrix::build_merged(const Mat &Q, const Mat &G, const NonsmoothEval &c, const int k, Mat &A)
    {
        Structure &T = *t_;
        const size_t nb = c.blocks.size();
        // match the blocks of this point to the structure (same index list), write their values to their positions
        std::vector<char> in_structure(nb, 0);
        std::fill(T.vals.begin(), T.vals.end(), 0.0);
        size_t n_new = 0;
        for (size_t b = 0; b < nb; ++b)
        {
            const HessianBlock &hb = c.blocks[b];
            const uint64_t h = hash_order(hb.indices.data(), hb.size);
            const Structure::Blk *found = nullptr;
            const auto rg = T.by_hash.equal_range(h);
            for (auto it = rg.first; it != rg.second && found == nullptr; ++it)
            {
                const Structure::Blk &B = T.blk[size_t(it->second)];
                if (B.m != hb.size)
                    continue;
                bool same = true;
                for (int i = 0; i < hb.size && same; ++i)
                    same = B.ord[size_t(i)] == hb.indices[i];
                if (same)
                    found = &B;
            }
            if (found == nullptr)
            {
                ++n_new;
                continue;
            }
            in_structure[b] = 1;
            const int m = hb.size;
            const int *P = &T.pos[size_t(found->off)];
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                    if (const int ps = P[i * m + j]; ps >= 0)
                        T.vals[size_t(ps)] += hb.H(i, j);
        }
        // W = K_R Q_R, A = G + Q_R^T W
        const int nR = int(T.R.size());
        T.QR.resize(nR, k);
        for (int i = 0; i < nR; ++i)
            gather(Q, T.R[size_t(i)], k, T.QR.row(i));
        {
            Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>> K(nR, nR, int(T.inner.size()), T.outer.data(), T.inner.data(), T.vals.data());
            T.W.noalias() = K * T.QR;
        }
        A = G;
        A.noalias() += T.QR.transpose() * T.W;
        // the blocks that are not in the structure, one by one
        if (n_new > 0)
        {
            Eigen::Matrix<double, HessianBlock::max_size, Eigen::Dynamic> LQ(HessianBlock::max_size, k);
            for (size_t b = 0; b < nb; ++b)
            {
                if (in_structure[b])
                    continue;
                const HessianBlock &hb = c.blocks[b];
                LQ.setZero();
                bool any = false;
                for (int i = 0; i < hb.size; ++i)
                    if (hb.indices[i] >= 0)
                    {
                        gather(Q, hb.indices[i], k, LQ.row(i));
                        any = true;
                    }
                if (any)
                    A.noalias() += LQ.topRows(hb.size).transpose() * (hb.H.topLeftCorner(hb.size, hb.size) * LQ.topRows(hb.size));
            }
        }
    }

    void ReducedMatrix::build(const Mat &Q, const Mat &G, const NonsmoothEval &c, const int k, Mat &A)
    {
        if (!c.local)
        {
            const Mat KQ = c.K * Q.leftCols(k);
            A = G + Q.leftCols(k).transpose() * KQ;
            return;
        }
        if (first_)
        {
            // the first point of the outer iteration decides the form of all its points
            first_ = false;
            merged_ = mode_ == "merged" || (mode_ == "auto" && int(c.blocks.size()) >= min_blocks_);
            if (merged_)
                make_structure(c, Q.rows());
        }
        if (merged_)
        {
            build_merged(Q, G, c, k, A);
            return;
        }
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
            add_blocks(Q, c, k, A, 0, nb);
        }
        else
        {
            std::vector<Mat> acc(nc);
            parallel::for_chunks(nb, nc, [&](const int ci, const Eigen::Index b0, const Eigen::Index b1) {
#ifdef EIGEN_USE_MKL
                const int mkl_threads = mkl_set_num_threads_local(1);
#endif
                acc[ci] = ci == 0 ? G : Mat(Mat::Zero(k, k));
                add_blocks(Q, c, k, acc[ci], b0, b1);
#ifdef EIGEN_USE_MKL
                mkl_set_num_threads_local(mkl_threads);
#endif
            });
            A = std::move(acc[0]);
            for (int i = 1; i < nc; ++i)
                A += acc[i];
        }
    }
} // namespace polysolve::nonlinear
