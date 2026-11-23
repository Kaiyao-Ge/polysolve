#pragma once

#include "../NonsmoothNewtonBase.hpp"

#include <polysolve/nonlinear/NonsmoothPorts.hpp>

#include <polysolve/linear/Solver.hpp>

namespace polysolve::nonlinear
{
    /// @brief The basis matrix B = H_s + proj(H_n) of an outer iteration (or proj(H_s) + proj(H_n)) with its pieces, its
    /// factorization and the Newton direction -B^-1 g.
    class BasisMatrix
    {
    public:
        using Vec = Eigen::VectorXd;
        using SpMat = StiffnessMatrix;

        BasisMatrix(const NonsmoothNewtonBase::Options &opts, NonsmoothNewtonBase::Stats &stats,
                    const json &solver_params, const json &linear_solver_params, spdlog::logger &logger);

        /// @brief Throws unless the linear solver is an LLT solver that reports indefinite matrices (checked on first use,
        /// so that a problem without the split can still select the strategy)
        void check_solver();

        /// @brief B = H_s + proj(H_n) and the per-direction pieces (Hnc, Hcp, gnc, gc0, Ec0, B) at x.
        void assemble_parts(Problem &p, const Vec &x);

        /// @brief B = H_s + proj(H_n) first; if its factorization fails, or p_N would need strong validity shortening, use
        /// B = proj(H_s) + proj(H_n) for this outer iteration and the model's quadratic part proj(H_s).
        /// @param h_edge Length scale of the nonsmooth term
        bool direction_projected_mode(Problem &p, const Vec &x, const Vec &g, Vec &d, const double h_edge);

        /// @brief Takes the change of the nonsmooth term's model into the pieces in place (the units moved from the smooth
        /// part to the nonsmooth term): Hnc, gnc, gc0 and Ec0. B and its factorization are not touched, so B != Hnc + Hcp from
        /// here on (reanchored). False, with nothing changed, if no unit was waiting.
        bool apply_delta(const NonsmoothDelta &delta);

        /// @brief Solves B u = rhs with the last factorization without touching the counters of the method (the selection's back-solves)
        bool solve_plain(const Vec &rhs, Vec &out);

        /// @brief Solves B u = rhs with the last factorization
        Vec backsolve(const Vec &rhs);

        // per direction
        SpMat Hnc, Hcp, B;
        Vec gnc, gc0;
        double Ec0 = 0;
        bool projected = false;
        bool reanchored = false; ///< the smooth part was changed in place after B was formed: B != Hnc + Hcp

    private:
        void factorize(const SpMat &H);
        bool same_pattern(const SpMat &H) const;
        bool try_solve(Problem &p, const SpMat &H, const Vec &g, Vec &d, const NormType nt);

        const NonsmoothNewtonBase::Options &opts_;
        NonsmoothNewtonBase::Stats &st_;
        spdlog::logger &logger_;
        double res_tol_ = 1e-5;
        std::unique_ptr<polysolve::linear::Solver> linear_;
        std::vector<SpMat::StorageIndex> pattern_outer_, pattern_inner_; ///< pattern of the last symbolic analysis
        Eigen::Index pattern_rows_ = -1;
        bool pattern_ok_ = false;
    };
} // namespace polysolve::nonlinear
