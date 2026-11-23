#include "SubspaceModel.hpp"
#include "../ParallelKernels.hpp"

#include <Eigen/Cholesky>

#include <cmath>

namespace polysolve::nonlinear
{
    SubspaceModel::SubspaceModel(const BasisMatrix &basis, ReducedMatrix &reduced, const int n_, const int cap_, const bool reduced_algebra, const bool row_major_q)
        : n(n_), cap(cap_), reduced_algebra_(reduced_algebra), row_major_(row_major_q), basis_(basis), reduced_(reduced)
    {
        if (row_major_)
            Qr.resize(n, cap);
        Q.resize(n, cap);
        HQ.resize(n, cap);
        G = Mat::Zero(cap, cap);
        ell = Vec::Zero(cap);
        cq = Vec::Zero(cap);
    }

    void SubspaceModel::add_column(const Vec &q)
    {
        const int j = k;
        Q.col(j) = q;
        parallel::spmv(basis_.Hnc, q, HQ.col(j));
        // one dot product per column, the columns split over the threads
        parallel::for_chunks(j + 1, parallel::chunks(j + 1, 2), [&](int, const Eigen::Index i0, const Eigen::Index i1) {
            for (Eigen::Index i = i0; i < i1; ++i)
            {
                const double v = Q.col(i).dot(HQ.col(j));
                G(i, j) = v;
                G(j, i) = v;
            }
        });
        ell[j] = q.dot(basis_.gnc);
        if (row_major_)
            for (Eigen::Index i = 0; i < q.size(); ++i)
                Qr(i, j) = q[i];
        ++k;
    }

    void SubspaceModel::apply_delta(const NonsmoothDelta &delta)
    {
        if (k == 0)
            return;
        const Mat dHQ = Mat(delta.dH * Q.leftCols(k));
        HQ.leftCols(k) -= dHQ;
        if (reduced_algebra_)
        {
            // the model gradient takes G z directly: G is formed again from the updated HQ (no accumulated rounding of the old G)
            Mat Gn = Q.leftCols(k).transpose() * HQ.leftCols(k);
            const Mat Gnt = Gn.transpose();
            G.topLeftCorner(k, k) = 0.5 * (Gn + Gnt);
        }
        else
        {
            Mat dG = Q.leftCols(k).transpose() * dHQ;
            const Mat dGt = dG.transpose();
            dG = 0.5 * (dG + dGt);
            G.topLeftCorner(k, k) -= dG;
        }
        ell.head(k) -= Q.leftCols(k).transpose() * delta.dg;
    }

    SubspaceModel::Eval SubspaceModel::eval(const Vec &z, const NonsmoothEval &c, bool deriv) const
    {
        Eval e;
        const Vec zk = z.head(k);
        const Mat Gk = Mat(G.topLeftCorner(k, k));
        const Vec Gz = Gk * zk;
        const double lin = ell.head(k).dot(zk), quad = 0.5 * zk.dot(Gz), cc = c.E - basis_.Ec0;
        e.m = lin + quad + cc;
        e.sumabs = std::abs(lin) + std::abs(quad) + (std::abs(c.E) + std::abs(basis_.Ec0));
        if (rr)
        {
            // the correction of the model gradient integrates to cq . z; m_off keeps the value continuous at the last re-read point
            const double cz = cq.head(k).dot(zk);
            e.m += cz + m_off;
            e.sumabs += std::abs(cz) + std::abs(m_off);
        }
        if (deriv)
        {
            if (reduced_algebra_)
            {
                // ell = Q^T g_s and G = Q^T H_s Q are kept in step with the in-place updates: only the rows of the nonsmooth
                // gradient that can be nonzero are summed, in the order the blocks first mention them (global path: its nonzeros)
                e.grad = ell.head(k) + Gz;
                const auto add_row = [&](const Eigen::Index r) {
                    const double gr = c.g[r];
                    if (gr == 0.0)
                        return;
                    if (row_major_)
                    {
                        const double *qr = &Qr(r, 0);
                        for (int j = 0; j < k; ++j)
                            e.grad[j] += qr[j] * gr;
                    }
                    else
                        for (int j = 0; j < k; ++j)
                            e.grad[j] += Q(r, j) * gr;
                };
                if (c.local)
                {
                    if (row_mark_.size() != std::size_t(c.g.size()))
                    {
                        row_mark_.assign(std::size_t(c.g.size()), 0);
                        row_epoch_ = 0;
                    }
                    const int ep = ++row_epoch_;
                    for (const auto &b : c.blocks)
                        for (int i = 0; i < b.size; ++i)
                            if (const int r = b.indices[i]; r >= 0 && row_mark_[r] != ep)
                            {
                                row_mark_[r] = ep;
                                add_row(r);
                            }
                }
                else
                    for (Eigen::Index i = 0; i < c.g.size(); ++i)
                        add_row(i);
            }
            else
            {
                const Vec Hp = HQ.leftCols(k) * zk;
                const Vec full = basis_.gnc + Hp + c.g;
                e.grad = Q.leftCols(k).transpose() * full;
            }
            if (rr)
                e.grad += cq.head(k);
            reduced_.set_row_major_q(row_major_ ? &Qr : nullptr);
            reduced_.build(Q, Gk, c, k, e.A);
            const Mat At = e.A.transpose();
            e.A = 0.5 * (e.A + At);
        }
        return e;
    }

    bool SubspaceModel::lm_step(const Mat &A, const Vec &grad, Vec &dz)
    {
        const double anorm = A.cwiseAbs().rowwise().sum().maxCoeff();
        const double sigma0 = 1e-5 * std::max(1.0, anorm);
        Mat As = A;
        for (int t = 0; t < 18; ++t)
        {
            const double sigma = t == 0 ? 0.0 : sigma0 * std::pow(10.0, t - 1);
            As = A;
            As.diagonal().array() += sigma;
            Eigen::LLT<Mat> llt(As);
            if (llt.info() != Eigen::Success)
                continue;
            dz = -llt.solve(grad);
            if (dz.allFinite() && grad.dot(dz) < 0)
                return true;
        }
        return false;
    }
} // namespace polysolve::nonlinear
