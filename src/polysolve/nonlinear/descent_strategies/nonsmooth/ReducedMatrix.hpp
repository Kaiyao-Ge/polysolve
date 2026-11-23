#pragma once

#include <polysolve/nonlinear/Problem.hpp>

#include <vector>

namespace polysolve::nonlinear
{
    /// @brief Nonsmooth term at a trial point y = x + p
    struct NonsmoothEval
    {
        Eigen::VectorXd y; ///< trial point
        double E = 0;      ///< value
        bool deriv = false;
        Eigen::VectorXd g;                ///< gradient
        StiffnessMatrix K;                ///< unprojected Hessian (global path)
        std::vector<HessianBlock> blocks; ///< unprojected Hessian blocks (local path)
        bool local = false;
    };

    /// @brief The matrix A = G + Q^T K Q of the reduced model, K the unprojected Hessian of the nonsmooth term at a trial
    /// point.
    class ReducedMatrix
    {
    public:
        using Mat = Eigen::MatrixXd;

        /// @brief A (before the symmetrization) from the k x k part G of the basis Gram matrix and the first k columns of Q
        void build(const Mat &Q, const Mat &G, const NonsmoothEval &c, const int k, Mat &A) const;
    };
} // namespace polysolve::nonlinear
