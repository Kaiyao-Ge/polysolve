#pragma once

#include "SubspaceModel.hpp"

#include <polysolve/nonlinear/NonsmoothPorts.hpp>

#include <limits>
#include <vector>

namespace polysolve::nonlinear
{
    /// @brief What the selection of the problem takes part in during a path.
    struct PathContext
    {
        FirstLayerPort *first_layer = nullptr; ///< null: a limiter only cuts the step
        SelectionPort *selection = nullptr;    ///< needed by Jg
        unsigned events = 60;                  ///< LimitEvent kinds (bit i) that promote units
        bool reread = false;                   ///< true gradient re-read at the inner points
        bool jgate = false;                    ///< Jg: the relative Jacobian limit on the exact units is switched by its effect
    };

    /// @brief What the last path did, for the handoff gate, Jg and the wall switch of the selection
    struct PathInfo
    {
        double E_acc = std::numeric_limits<double>::quiet_NaN(); ///< true energy at the accepted point
        long trunc[6] = {0, 0, 0, 0, 0, 0};                      ///< cuts by guard, valid, ftb, ccd, jfloor, jbound
        bool jg_end = false, jg_retest = false, jg_event = false;
    };

    /// @brief The protected path of an outer iteration: minimizes the reduced model with Levenberg-Marquardt steps in the
    /// subspace of a SubspaceModel that grows with B^-1 (model residual) columns, limits every step (segment guard,
    /// validity, fraction-to-boundary, max_step_size, Jacobian limits), and accepts the end point of the polyline (or a
    /// retraction along it) by its true decrease.
    class PathRunner
    {
    public:
        using Vec = Eigen::VectorXd;
        using Mat = Eigen::MatrixXd;
        using SpMat = StiffnessMatrix;

        PathRunner(const NonsmoothNewtonBase::Options &opts, NonsmoothNewtonBase::Stats &stats, BasisMatrix &basis, spdlog::logger &logger)
            : opts_(opts), st_(stats), basis_(basis), reduced_(opts.reduced_matrix, opts.merged_min_blocks), logger_(logger)
        {
        }

        /// @brief The path from x with the Newton direction pN: true on success, with the end point in step (x + step) and
        /// the number of accepted inner steps in n_inner. noise_stop tells whether the path stopped at the noise floor.
        /// @param E0 true energy at x
        /// @param ctx The first layer: if a fraction-to-boundary, max_step_size or Jacobian cut leaves new units (its port), they are
        /// taken into the model in place and the polyline restarts at the current point (at most Options::lift_max_rounds times per path).
        bool run(Problem &p, const Vec &x, const Vec &g, double E0, const Vec &pN, Vec &step, int &n_inner, const PathContext &ctx);

        /// @brief A Jacobian limit is on but the problem gives NaN for it: reported once per solve.
        void jacobian_missing(bool &reported, const char *limit, const char *callback);

        /// @brief Start of a minimize: forgets the Jacobian reference
        void reset();

        // per minimize
        Vec Mred;               ///< metric diagonal
        double h_edge = std::numeric_limits<double>::quiet_NaN(); ///< length scale of the nonsmooth term
        double j_ref = std::numeric_limits<double>::quiet_NaN();  ///< Jacobian floor: min J at the first iterate of this solve with a computed direction
        double min_j_x = std::numeric_limits<double>::quiet_NaN(); ///< Jacobian floor: min J at the current outer iterate
        bool j_floor_missing = false, j_bound_missing = false; ///< the missing Jacobians were reported in this solve

        PathInfo info; ///< the last path
        bool noise_stop = false; ///< the last path stopped at the noise floor or below the resolution of the true energy

    private:
        double mnorm2(const Vec &v) const;
        int jacobian_violation(Problem &p, const Vec &x, const Vec &y, const double jfloor);
        NonsmoothEval eval(Problem &p, const Vec &y, bool deriv, NonsmoothEval *reuse = nullptr);
        double true_energy(Problem &p, const Vec &y);

        const NonsmoothNewtonBase::Options &opts_;
        NonsmoothNewtonBase::Stats &st_;
        BasisMatrix &basis_;
        ReducedMatrix reduced_;
        spdlog::logger &logger_;

        Vec last_built_;                         ///< point of the last nonsmooth_update done by eval (empty: unknown)
        std::vector<HessianBlock> spare_blocks_; ///< block buffer of a replaced state, reused by eval
    };
} // namespace polysolve::nonlinear
