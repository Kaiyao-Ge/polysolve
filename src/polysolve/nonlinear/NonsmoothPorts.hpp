#pragma once

#include "Problem.hpp"

#include <deque>
#include <functional>
#include <limits>

namespace polysolve::nonlinear
{
    /// @brief A limiter of the protected path cut (or ended) an inner step.
    struct LimitEvent
    {
        enum class Kind
        {
            guard,       ///< long-segment guard
            valid,       ///< validity shortening of the segment end
            ftb,         ///< linearized fraction-to-boundary
            ccd,         ///< max_step_size on the segment
            jacobian_floor,
            jacobian_bound
        } kind;
        const Problem::TVector &x; ///< outer iterate of the path
        int inner;                 ///< accepted inner steps so far
        const Problem::TVector *a = nullptr; ///< segment start (ftb, ccd) or outer iterate (Jacobian limits)
        const Problem::TVector *b = nullptr; ///< segment end (ftb, ccd) or end point of the step (Jacobian limits)
        double ftb_tau = 0;                  ///< ftb: the fraction-to-boundary parameter
        double jacobian_floor = -std::numeric_limits<double>::infinity(); ///< Jacobian limits: the floor of this outer iteration (-inf: off)
        double jacobian_bound = 0;           ///< Jacobian limits: the bound of the volume ratio (0: off)
        bool promote = true;                 ///< false: the units behind the cut are only to be recorded (the rounds of the first layer are used up)
    };

    /// @brief Change of the nonsmooth term's quadratic model, applied in place to the basis matrix and the subspace model.
    struct NonsmoothDelta
    {
        Problem::THessian dH;
        Problem::TVector dg;
        double dE = 0;
        int n = 0; ///< number of units changed
    };

    /// @brief Units of the smooth part that the protected path promotes to the nonsmooth term when a limiter cuts a step.
    class FirstLayerPort
    {
    public:
        virtual ~FirstLayerPort() = default;
        /// @brief Makes the units behind the event exact (or, if event.promote is false, only records them for the selection of the next
        /// outer iteration); returns the number of units that became exact.
        virtual int identify(const LimitEvent &event) = 0;
        /// @brief The change of the model that the units recorded since the last call cause.
        virtual bool take_delta(const Problem::TVector &x, const bool projected, NonsmoothDelta &delta) = 0;
    };

    /// @brief Jg: whether the relative Jacobian limit applies to the units that are exact. A limit that ended a path pays if
    /// the next outer iteration of the solve has no limit event itself, else it does not pay (also pays if the solve ends
    /// first). The score is the sum of the last `window` outcomes (+1 / -1); a negative score switches the limit off for the
    /// exact units. While it is off every `retest`-th outer iteration with such an event enforces the limit from that step on
    /// (the path ends there) and its outcome enters the score. The state lives across solves (the session of the problem).
    struct JGate
    {
        int window = 4;
        int retest = 5;
        bool enabled = true; ///< the limit applies to the exact units
        std::deque<int> scores;
        int dis_events = 0; ///< outer iterations with a limit event while disabled (since the signal was disabled)
        int applied = -1; ///< the value last given to the problem (-1: none in this solve yet)
        bool pending = false;

        int score() const
        {
            int s = 0;
            for (const int v : scores)
                s += v;
            return s;
        }
        /// @brief Whether the limit should apply to the exact units at the start of the next outer iteration
        int want() const { return enabled ? 1 : 0; }
        void resolve(const bool pays)
        {
            if (!pending)
                return;
            pending = false;
            const bool was = enabled;
            scores.push_back(pays ? 1 : -1);
            while (int(scores.size()) > window)
                scores.pop_front();
            enabled = score() >= 0;
            if (enabled != was)
                dis_events = 0;
        }
        /// @brief A solve starts: a pending event whose solve converged pays
        void solve_start()
        {
            resolve(true);
            applied = -1;
        }
        /// @brief End of the path of an outer iteration: resolves the pending event of the previous outer iteration with this
        /// one's limit event, then registers this one's limit end
        void outer_end(const bool event, const bool end)
        {
            if (pending)
                resolve(!event);
            if (end)
                pending = true;
        }
    };

    /// @brief Choice of the units that are modelled exactly (the blind selection).
    class SelectionPort
    {
    public:
        struct Context
        {
            const Problem::TVector &x;
            const Problem::TVector &grad;
            double res2;       ///< g . B^-1 g
            /// back-solve with the factorization of B (the counters of the strategy stay untouched)
            std::function<bool(const Problem::TVector &, Problem::TVector &)> solve;
            bool projected;    ///< B holds the projected smooth Hessian
            double jacobian_bound;
            double jacobian_floor; ///< -inf: off
        };
        virtual ~SelectionPort() = default;
        /// @brief The second and third layer at the start of an outer iteration (after the factorization of B): the units
        /// that became exact, as the change of the model; returns their number.
        virtual int select(const Context &context, NonsmoothDelta &delta) = 0;
        /// @brief False where the re-read is switched off for this outer iteration
        virtual bool reread_allowed() const { return true; }
        virtual JGate &jgate() = 0;
        /// @brief Jg: whether the relative Jacobian limit (bound) applies to the exact units too (1) or only to the others (0)
        virtual void jratio_exact(const int on) = 0;
        /// @brief Jg: the smallest ratio J_e(y) / J_e(x) over the exact units (NaN: none / not available); the term's state is not changed
        virtual double min_ratio_exact(const Problem::TVector &x, const Problem::TVector &y) = 0;
        /// @brief Called at the end of every outer iteration: the cuts of its path by guard, valid, ftb, ccd, jfloor, jbound and the
        /// number of inner steps it accepted
        virtual void outer_end(const long *cuts, const long inner) = 0;
    };

    /// @brief A round of the repair of a basis matrix whose factorization failed.
    struct RepairQuery
    {
        const Problem::TVector &x; ///< the outer iterate (the anchor of B)
        int round;                 ///< 1, 2, ...
        long fail_index;           ///< the unknown at which the last factorization stopped (-1: not known)
        int radius;                ///< the radius (in the graph of the units' nodes) of the first round around a failure node
    };

    /// @brief Corrections of the basis matrix B that the problem knows (static correction of the lifted units, repair).
    class BasisPort
    {
    public:
        virtual ~BasisPort() = default;
        /// @brief Static correction of B for the units of the first layer that the model keeps frozen: dB = proj(K) - K
        virtual bool correction(const Problem::TVector &x, const bool projected, Problem::THessian &dB) = 0;
        /// @brief Local repair of an indefinite B: the frozen units with negative curvature around the failure of the factorization get
        /// their projected block (B + dB is factorized again). dB is the total change of B of all rounds so far; false: no repair possible
        /// (nothing new to project), the caller then projects the whole smooth part.
        virtual bool repair(const RepairQuery &query, Problem::THessian &dB, bool &final_round) = 0;
    };

    /// @brief Sink of the strategy's log rows.
    class LogPort
    {
    public:
        virtual ~LogPort() = default;
        virtual bool on() const = 0;
        virtual void put(const json &row) = 0;
    };
} // namespace polysolve::nonlinear
