#pragma once

#include "BasisMatrix.hpp"
#include "ReducedMatrix.hpp"

namespace polysolve::nonlinear
{
    /// @brief The reduced-space model of one outer iteration: M-orthonormal columns Q, HQ = H_s Q, G = Q^T H_s Q,
    /// ell = Q^T g_s, with H_s and g_s from the basis matrix, and the model m(z) = ell^T z + 1/2 z^T G z + n(x + Q z) - n(x).
    class SubspaceModel
    {
    public:
        using Vec = Eigen::VectorXd;
        using Mat = Eigen::MatrixXd;

        struct Eval
        {
            double m = 0;
            double sumabs = 0;
            Vec grad;
            Mat A;
        };

        SubspaceModel(const BasisMatrix &basis, const ReducedMatrix &reduced, const int n, const int cap);

        /// @brief Appends the column q (M-orthonormal to the others)
        void add_column(const Vec &q);

        /// @brief Takes the change of the smooth part (units moved to the nonsmooth term) into HQ, G and ell in place
        void apply_delta(const NonsmoothDelta &delta);

        /// @brief The model at z (and, if deriv, its gradient and the reduced matrix A) with the nonsmooth term c at x + Q z
        Eval eval(const Vec &z, const NonsmoothEval &c, bool deriv) const;

        /// @brief Levenberg-Marquardt step on the reduced model: sigma = 0 first, then sigma0 10^(t-1)
        static bool lm_step(const Mat &A, const Vec &grad, Vec &dz);

        int n = 0, k = 0, cap = 0;
        Mat Q, HQ, G;
        Vec ell;
        bool rr = false;  ///< re-read: the model has the correction cq / m_off
        Vec cq;           ///< Q^T cR, the correction of the model gradient
        double m_off = 0; ///< model value offset (continuity at the re-read point)

    private:
        const BasisMatrix &basis_;
        const ReducedMatrix &reduced_;
    };
} // namespace polysolve::nonlinear
