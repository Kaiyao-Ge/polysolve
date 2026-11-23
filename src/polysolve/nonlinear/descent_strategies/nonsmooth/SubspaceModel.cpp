#include "SubspaceModel.hpp"
#include "../ParallelKernels.hpp"

#include <Eigen/Cholesky>

#include <cmath>

namespace polysolve::nonlinear
{
    SubspaceModel::SubspaceModel(const BasisMatrix &basis, const ReducedMatrix &reduced, const int n_, const int cap_)
        : n(n_), cap(cap_), basis_(basis), reduced_(reduced)
    {
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
        ++k;
    }

    void SubspaceModel::apply_delta(const NonsmoothDelta &delta)
    {
        if (k == 0)
            return;
        const Mat dHQ = Mat(delta.dH * Q.leftCols(k));
        HQ.leftCols(k) -= dHQ;
        Mat dG = Q.leftCols(k).transpose() * dHQ;
        const Mat dGt = dG.transpose();
        dG = 0.5 * (dG + dGt);
        G.topLeftCorner(k, k) -= dG;
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
            const Vec Hp = HQ.leftCols(k) * zk;
            const Vec full = basis_.gnc + Hp + c.g;
            e.grad = Q.leftCols(k).transpose() * full;
            if (rr)
                e.grad += cq.head(k);
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
