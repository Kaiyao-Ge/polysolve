#include "NonsmoothNewtonBase.hpp"
#include "nonsmooth/PathRunner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace polysolve::nonlinear
{
    namespace
    {
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
        constexpr double EPS = std::numeric_limits<double>::epsilon();

        template <typename T>
        T value_or(const json &j, const char *key, const T &def)
        {
            return j.is_object() && j.contains(key) ? j[key].get<T>() : def;
        }

        /// solver_params/norm_type as Solver::set_line_search reads it for the line search's switch to comparing gradient
        /// norms (unknown names map to Euclidean, as there); the strategy's own norm type if the entry is missing
        NormType line_search_norm(const json &solver_params, const NormType fallback)
        {
            if (!solver_params.contains("norm_type") || !solver_params["norm_type"].is_string())
                return fallback;
            const std::string s = solver_params["norm_type"];
            return s == "L2" ? NormType::L2 : s == "Linf" ? NormType::Linf : NormType::EUCLIDEAN;
        }

        void check_choice(const char *key, const std::string &value, const std::vector<std::string> &options)
        {
            if (std::find(options.begin(), options.end(), value) == options.end())
                throw std::runtime_error(std::string("NonsmoothNewton: ") + key + " is not valid, got " + value);
        }

        /// A method part with the values auto, on and off; auto is off for the energy split.
        void check_part(const char *key, const std::string &value)
        {
            check_choice(key, value, {"auto", "on", "off"});
        }
    } // namespace

    NonsmoothNewtonBase::Options NonsmoothNewtonBase::Options::from_json(const json &solver_params)
    {
        // parameter block NonsmoothNewton (or the parameters themselves for an entry of a strategy list)
        const json &p = solver_params.contains("NonsmoothNewton") ? solver_params["NonsmoothNewton"] : solver_params;
        Options o;
        o.selection = value_or(p, "selection", o.selection);
        o.max_inner = value_or(p, "max_inner", o.max_inner);
        o.relative_stop = value_or(p, "relative_stop", o.relative_stop);
        o.ftb_tau = value_or(p, "fraction_to_boundary", o.ftb_tau);
        o.rho_min = value_or(p, "rho_min", o.rho_min);
        o.max_columns = value_or(p, "max_columns", o.max_columns);
        o.handoff = value_or(p, "handoff", o.handoff);
        o.resolution = value_or(p, "resolution", o.resolution);
        if (p.is_object() && p.contains("guard"))
        {
            const json &g = p["guard"];
            o.guard = value_or(g, "enabled", o.guard);
            o.guard_c = value_or(g, "c", o.guard_c);
            o.guard_abs = value_or(g, "abs", o.guard_abs);
            o.guard_check = value_or(g, "check", o.guard_check);
        }
        if (p.is_object() && p.contains("hand_back"))
        {
            const json &h = p["hand_back"];
            o.hand_back = value_or(h, "enabled", o.hand_back);
            o.hand_back_energy = value_or(h, "energy", o.hand_back_energy);
            o.hand_back_contraction = value_or(h, "contraction", o.hand_back_contraction);
        }
        if (p.is_object() && p.contains("jacobian"))
        {
            const json &j = p["jacobian"];
            o.jacobian_bound = value_or(j, "bound", o.jacobian_bound);
            o.jacobian_floor = value_or(j, "floor", o.jacobian_floor);
            o.exact_gate = value_or(j, "exact_gate", o.exact_gate);
            o.exact_gate_window = value_or(j, "exact_gate_window", o.exact_gate_window);
            o.exact_gate_retest = value_or(j, "exact_gate_retest", o.exact_gate_retest);
        }
        if (p.is_object() && p.contains("debug"))
        {
            const json &d = p["debug"];
            o.dedupe = value_or(d, "dedupe", o.dedupe);
            o.fast_validity = value_or(d, "fast_validity", o.fast_validity);
        }
        if (p.is_object() && p.contains("path"))
            o.path_always = value_or(p["path"], "always", o.path_always);
        if (p.is_object() && p.contains("reread"))
            o.reread = value_or(p["reread"], "enabled", o.reread);
        if (p.is_object() && p.contains("first_layer"))
        {
            const json &f = p["first_layer"];
            o.first_layer = value_or(f, "enabled", o.first_layer);
            o.lift_model = value_or(f, "lift_model", o.lift_model);
            o.lift_basis = value_or(f, "lift_basis", o.lift_basis);
            o.lift_max_rounds = value_or(f, "max_rounds", o.lift_max_rounds);
            o.restart_continuity = value_or(f, "restart_continuity", o.restart_continuity);
            if (f.is_object() && f.contains("events"))
            {
                o.first_layer_events = 0;
                for (const auto &e : f["events"])
                {
                    const std::string name = e.get<std::string>();
                    if (name == "ftb")
                        o.first_layer_events |= 1u << 2;
                    else if (name == "ccd")
                        o.first_layer_events |= 1u << 3;
                    else if (name == "jfloor")
                        o.first_layer_events |= 1u << 4;
                    else if (name == "jbound")
                        o.first_layer_events |= 1u << 5;
                    else
                        throw std::runtime_error("NonsmoothNewton: first_layer/events must be among jfloor, jbound, ftb, ccd, got " + name);
                }
            }
        }
        if (p.is_object() && p.contains("handoff_gate"))
        {
            const json &h = p["handoff_gate"];
            o.handoff_gate = value_or(h, "enabled", o.handoff_gate);
            o.handoff_gate_factor = value_or(h, "factor", o.handoff_gate_factor);
            o.handoff_gate_max_refusals = value_or(h, "max_refusals", o.handoff_gate_max_refusals);
        }
        if (p.is_object() && p.contains("model"))
        {
            const json &m = p["model"];
            o.reduced_algebra = value_or(m, "reduced_algebra", o.reduced_algebra);
            o.row_major_q = value_or(m, "row_major_q", o.row_major_q);
        }
        if (p.is_object() && p.contains("reduced_matrix"))
        {
            const json &r = p["reduced_matrix"];
            o.reduced_matrix = value_or(r, "mode", o.reduced_matrix);
            o.merged_min_blocks = value_or(r, "merged_min_blocks", o.merged_min_blocks);
            o.reduced_matrix_check = value_or(r, "check", o.reduced_matrix_check);
        }
        if (p.is_object() && p.contains("basis_repair"))
        {
            const json &b = p["basis_repair"];
            o.basis_repair = value_or(b, "mode", o.basis_repair);
            o.basis_repair_max_rounds = value_or(b, "max_rounds", o.basis_repair_max_rounds);
            o.basis_repair_radius = value_or(b, "radius", o.basis_repair_radius);
        }
        if (o.max_inner < 1)
            throw std::runtime_error("NonsmoothNewton: max_inner must be >= 1");
        if (o.ftb_tau < 0 || o.ftb_tau >= 1)
            throw std::runtime_error("NonsmoothNewton: fraction_to_boundary must be in [0, 1)");
        if (!(o.relative_stop >= 0) || !std::isfinite(o.relative_stop))
            throw std::runtime_error("NonsmoothNewton: relative_stop must be finite and >= 0");
        // rho_min <= 0 would accept a true energy increase
        if (!(o.rho_min > 0 && o.rho_min < 1))
            throw std::runtime_error("NonsmoothNewton: rho_min must be in (0, 1), got " + std::to_string(o.rho_min));
        if (o.max_columns < 1)
            throw std::runtime_error("NonsmoothNewton: max_columns must be >= 1");
        if (o.handoff != "off" && o.handoff != "first" && o.handoff != "any")
            throw std::runtime_error("NonsmoothNewton: handoff must be off, first or any, got " + o.handoff);
        if (!(o.guard_c > 0) || !(o.guard_abs > 0) || !(o.guard_check >= 0)
            || !std::isfinite(o.guard_c) || !std::isfinite(o.guard_abs) || !std::isfinite(o.guard_check))
            throw std::runtime_error("NonsmoothNewton: guard/c and guard/abs must be finite and > 0, guard/check finite and >= 0");
        if (!(o.hand_back_energy >= 0) || !(o.hand_back_contraction >= 1)
            || !std::isfinite(o.hand_back_energy) || !std::isfinite(o.hand_back_contraction))
            throw std::runtime_error("NonsmoothNewton: hand_back/energy must be finite and >= 0, hand_back/contraction finite and >= 1");
        if (!(o.jacobian_bound >= 0 && o.jacobian_bound < 1))
            throw std::runtime_error("NonsmoothNewton: jacobian/bound must be in [0, 1), got " + std::to_string(o.jacobian_bound));
        if (!(o.jacobian_floor >= 0) || !std::isfinite(o.jacobian_floor))
            throw std::runtime_error("NonsmoothNewton: jacobian/floor must be finite and >= 0");
        if (o.selection != "energy_split" && o.selection != "blind")
            throw std::runtime_error("NonsmoothNewton: selection must be energy_split or blind, got " + o.selection);
        // the method parts of the later stages: only the settings that give the energy split of this version are available
        check_part("path/always", o.path_always);
        check_part("reread/enabled", o.reread);
        check_part("first_layer/enabled", o.first_layer);
        check_part("handoff_gate/enabled", o.handoff_gate);
        check_choice("first_layer/lift_model", o.lift_model, {"exact", "frozen", "frozen_projected"});
        check_choice("first_layer/lift_basis", o.lift_basis, {"projected", "raw"});
        if (o.lift_model == "frozen" && o.lift_basis == "raw")
            throw std::runtime_error("NonsmoothNewton: first_layer/lift_model frozen with lift_basis raw does not lift anything");
        check_choice("jacobian/exact_gate", o.exact_gate, {"auto", "off", "gate"});
        check_choice("reduced_matrix/mode", o.reduced_matrix, {"per_block", "auto", "merged"});
        check_choice("basis_repair/mode", o.basis_repair, {"off", "r_ad"});
        if (o.reduced_matrix != "per_block" || o.basis_repair != "off"
            || o.reduced_algebra || o.row_major_q || o.reduced_matrix_check)
            throw std::runtime_error("NonsmoothNewton: model/*, reduced_matrix/* and basis_repair/* other than their defaults are not available yet");
        if (o.lift_max_rounds < 1 || o.exact_gate_window < 1 || o.exact_gate_retest < 1 || o.handoff_gate_max_refusals < 1
            || o.merged_min_blocks < 1 || o.basis_repair_max_rounds < 1 || o.basis_repair_radius < 1)
            throw std::runtime_error("NonsmoothNewton: first_layer/max_rounds, jacobian/exact_gate_window, jacobian/exact_gate_retest, handoff_gate/max_refusals, reduced_matrix/merged_min_blocks, basis_repair/max_rounds and basis_repair/radius must be >= 1");
        if (!(o.handoff_gate_factor > 0))
            throw std::runtime_error("NonsmoothNewton: handoff_gate/factor must be > 0");
        return o;
    }

    // =================================================================================================================
    struct NonsmoothNewtonBase::Impl
    {
        /// outcome of the last take_step that went to the line search (undone if the line search fails)
        struct Pending
        {
            bool set = false;
            bool active = false;
            int inner = 0;
            long *kind = nullptr;
        };

        enum class Handoff
        {
            none,
            noise,   ///< after a noise-floor / resolution stop
            gradient ///< on a small gradient
        };

        Stats st;
        BasisMatrix basis;
        PathRunner runner;
        Pending pending;
        /// norm of the handoff test on a small gradient: the line search's (solver_params/norm_type), so that the handoff
        /// happens where the line search would switch to comparing gradient norms
        const NormType norm_type;
        double use_grad_norm_tol = -1; ///< line_search/use_grad_norm_tol of the solver (handoff on a small gradient)

        // per minimize
        bool fresh = true;
        bool handed_off = false;  ///< the rest of this solve is left to the Newton chain
        bool handing_off = false; ///< the last direction request was declined by the handoff (not a failure)
        Handoff handoff_kind = Handoff::none; ///< the last handoff of this solve
        bool no_grad_handoff = false;         ///< after a hand-back: no handoff on a small gradient for the rest of this solve

        // per direction
        bool use_B = false;

        // handoff gate and Jg
        int gate_refusals = 0;   ///< consecutive refusals of this solve without an accepted decrease larger than the energy resolution
        bool jg_solve_started = false; ///< Jg: the solve start was reported to the gate

        Impl(const Options &o, const json &solver_params, const json &linear_solver_params, spdlog::logger &logger, const NormType nt)
            : basis(o, st, solver_params, linear_solver_params, logger),
              runner(o, st, basis, logger),
              norm_type(line_search_norm(solver_params, nt))
        {
            if (solver_params.contains("line_search") && solver_params["line_search"].contains("use_grad_norm_tol"))
                use_grad_norm_tol = solver_params["line_search"]["use_grad_norm_tol"];
        }

        void init_minimize(Problem &p)
        {
            if (!fresh)
                return;
            // checked on first use, so that a problem without the split can still select this solver
            basis.check_solver();
            fresh = false;
            p.metric_diagonal(runner.Mred);
            runner.h_edge = p.nonsmooth_length_scale();
        }

        /// Jacobian floor of the outer iteration at x (-inf: off): jacobian_floor * J_ref, or min J at x where x is below it
        double current_jfloor(const Options &o) const
        {
            if (o.jacobian_floor > 0 && runner.j_ref > 0 && std::isfinite(runner.min_j_x))
                return runner.min_j_x >= o.jacobian_floor * runner.j_ref ? o.jacobian_floor * runner.j_ref : runner.min_j_x;
            return -std::numeric_limits<double>::infinity();
        }

        void record(long &kind, bool active, int inner)
        {
            ++kind;
            if (active)
                ++st.active_iters;
            st.inner += inner;
            pending = Pending{true, active, inner, &kind};
        }

        void undo_pending()
        {
            if (!pending.set)
                return;
            --*pending.kind;
            if (pending.active)
                --st.active_iters;
            st.inner -= pending.inner;
            pending = Pending();
        }
    };

    // =================================================================================================================
    NonsmoothNewtonBase::NonsmoothNewtonBase(const json &solver_params,
                                             const json &linear_solver_params,
                                             const double characteristic_length,
                                             spdlog::logger &logger,
                                             const NormType norm_type)
        : Superclass(solver_params, characteristic_length, logger),
          opts_(Options::from_json(solver_params))
    {
        impl_ = std::make_unique<Impl>(opts_, solver_params, linear_solver_params, logger, norm_type);
    }

    NonsmoothNewtonBase::~NonsmoothNewtonBase() = default;

    bool NonsmoothNewtonBase::compute_update_direction(Problem &objFunc, const TVector &x, const TVector &grad, TVector &direction)
    {
        Impl &m = *impl_;
        SelectionPort *const sel = selection_port(objFunc);
        if (sel != nullptr && opts_.exact_gate_on())
        {
            JGate &jg = sel->jgate();
            jg.window = opts_.exact_gate_window;
            jg.retest = opts_.exact_gate_retest;
            if (!m.jg_solve_started)
            {
                jg.solve_start(); // a solve starts: a pending event whose solve converged pays
                m.jg_solve_started = true;
            }
            const int w = jg.want();
            if (w != jg.applied)
            {
                sel->jratio_exact(w);
                jg.applied = w;
            }
        }
        m.pending = Impl::Pending();
        m.use_B = false;
        // no split (e.g. a full-size phase): leave the iteration to the Newton chain that follows
        if (!objFunc.has_nonsmooth_term())
            return false;
        // handoff on a small gradient, where the line search compares gradient norms (checked before any factorization)
        if (opts_.handoff != "off" && !m.handed_off && !m.no_grad_handoff
            && objFunc.grad_norm(grad, m.norm_type) < m.use_grad_norm_tol * objFunc.grad_norm_rescaling(m.norm_type))
        {
            m.handed_off = true;
            m.handoff_kind = Impl::Handoff::gradient;
            ++m.st.handoff_grad;
        }
        if (m.handed_off)
        {
            ++m.st.handoff_iters;
            m.handing_off = true;
            return false;
        }
        m.init_minimize(objFunc);
        // Jacobian floor: min J at x; its first value in the solve is the reference J_ref
        if (opts_.jacobian_floor > 0)
        {
            m.runner.min_j_x = objFunc.smooth_min_jacobian(x);
            if (!std::isfinite(m.runner.j_ref))
                m.runner.j_ref = m.runner.min_j_x;
            if (!std::isfinite(m.runner.min_j_x))
                m.runner.jacobian_missing(m.runner.j_floor_missing, "floor", "smooth_min_jacobian");
        }
        bool ok = false;
        try
        {
            m.basis.assemble_parts(objFunc, x);
            ok = m.basis.direction_projected_mode(objFunc, x, grad, direction, m.runner.h_edge);
        }
        catch (const std::runtime_error &e)
        {
            m_logger.debug("[{}] direction failed: {}", name(), e.what());
            ok = false;
        }
        if (!ok)
        {
            objFunc.solution_changed(x);
            return false;
        }
        m.use_B = true;
        if (sel != nullptr)
            select_units(objFunc, *sel, x, grad, direction);
        return true;
    }

    void NonsmoothNewtonBase::select_units(Problem &objFunc, SelectionPort &sel, const TVector &x, const TVector &grad, const TVector &direction)
    {
        Impl &m = *impl_;
        SelectionPort::Context ctx{x, grad, -grad.dot(direction), nullptr, m.basis.projected, opts_.jacobian_bound, m.current_jfloor(opts_)};
        ctx.solve = [&](const TVector &rhs, TVector &out) { return m.basis.solve_plain(rhs, out); };
        NonsmoothDelta delta;
        int n = 0;
        try
        {
            n = sel.select(ctx, delta);
        }
        catch (const std::runtime_error &e)
        {
            m_logger.debug("[{}] selection failed: {}", name(), e.what());
            n = 0;
        }
        ++m.st.select_calls;
        m.st.selected += n;
        if (n > 0)
            m.basis.apply_delta(delta);
    }

    bool NonsmoothNewtonBase::take_step(Problem &objFunc, const TVector &x, const double energy, const TVector &grad,
                                        const TVector &direction, TVector &step)
    {
        Impl &m = *impl_;
        if (!m.use_B)
        {
            m.record(m.st.ls_iters, false, 0);
            return false;
        }
        const bool active = opts_.path_always_on() || objFunc.nonsmooth_active_size() > 0;
        int n_inner = 0;
        m.runner.noise_stop = false;
        m.runner.info = PathInfo();
        SelectionPort *const sel = selection_port(objFunc);
        PathContext ctx;
        ctx.selection = sel;
        ctx.events = opts_.first_layer_events;
        ctx.jgate = sel != nullptr && opts_.exact_gate_on();
        ctx.reread = opts_.reread_on() && (sel == nullptr || sel->reread_allowed());
        if (opts_.first_layer_on())
        {
            ctx.first_layer = objFunc.nonsmooth_first_layer(); // none in a phase that the problem has no first layer for (e.g. full size)
        }
        bool took_path = active && grad.dot(direction) < 0 && m.runner.run(objFunc, x, grad, energy, direction, step, n_inner, ctx);
        const PathInfo &pi = m.runner.info;
        if (ctx.jgate && active)
            sel->jgate().outer_end(pi.jg_event, took_path && pi.jg_end);
        // the true energy no longer resolves the path: the Newton chain finishes the solve from the next iteration. With the
        // handoff gate the handoff waits while the energy is still far above its resolution and the path made progress.
        {
            const double e_res = 32 * std::numeric_limits<double>::epsilon() * std::abs(energy);
            const bool progress = took_path && std::isfinite(pi.E_acc) && (energy - pi.E_acc) > e_res;
            const bool would = opts_.handoff != "off" && !m.handed_off && m.runner.noise_stop && (opts_.handoff == "any" || n_inner == 0);
            const double half = 0.5 * -grad.dot(direction); // 1/2 g . B^-1 g at the anchor
            const bool high = std::isfinite(half) && half > opts_.handoff_gate_factor * e_res;
            bool do_handoff = would;
            const bool gate = opts_.handoff_gate_on();
            if (would && gate && high)
            {
                if (progress)
                {
                    m.gate_refusals = 0;
                    do_handoff = false;
                }
                else if (m.gate_refusals < opts_.handoff_gate_max_refusals)
                {
                    ++m.gate_refusals;
                    do_handoff = false;
                }
                // else: no progress and the refusals are used up: hand off as before
            }
            else if (progress)
                m.gate_refusals = 0;
            if (do_handoff)
            {
                m.handed_off = true;
                m.handoff_kind = Impl::Handoff::noise;
                ++m.st.handoff_noise;
            }
        }
        if (sel != nullptr)
            sel->outer_end(pi.trunc, n_inner);
        if (took_path)
        {
            m.record(m.st.path_iters, active, n_inner);
            m.pending = Impl::Pending();
            return true;
        }
        // restore the nonsmooth state at x before the line search
        objFunc.solution_changed(x);
        m.record(active ? m.st.path_fail : m.st.ls_iters, active, n_inner);
        return false;
    }

    bool NonsmoothNewtonBase::handle_error()
    {
        // a direction declined by the handoff is not a failure: move on to the Newton chain
        if (impl_->handing_off)
        {
            impl_->handing_off = false;
            return false;
        }
        ++impl_->st.failed;
        impl_->undo_pending(); // the line search on p_N failed: the iteration does not count
        return false;
    }

    void NonsmoothNewtonBase::chain_step_result(const int strategy, const double rate, const double energy_before, const double energy_after,
                                                const double grad_norm_before, const double grad_norm_after, const double grad_tol)
    {
        // only after a handoff on a small gradient (after a noise-floor stop the true energy no longer resolves the path)
        Impl &m = *impl_;
        if (!opts_.hand_back || !m.handed_off || m.handoff_kind != Impl::Handoff::gradient)
            return;
        // the step left the solve unconverged and removed energy the acceptance test of the path can resolve
        const double resolution = 32 * EPS * std::abs(energy_before);
        if (!(grad_norm_after > grad_tol) || !(energy_before - energy_after > opts_.hand_back_energy * resolution))
            return;
        // and Newton is not in its fast local regime: a full step taken by Newton itself that contracts the gradient norm
        const bool slow = rate < 1 || strategy >= 2 || !(grad_norm_before >= opts_.hand_back_contraction * grad_norm_after);
        if (!slow)
            return;
        // the method takes the next iteration the solver gives to it; a gradient handoff would decline it again at once
        m.handed_off = false;
        m.no_grad_handoff = true;
        ++m.st.hand_backs;
    }

    void NonsmoothNewtonBase::reset_times()
    {
        impl_->st = Stats();
        impl_->pending = Impl::Pending();
        impl_->fresh = true;
        impl_->handed_off = false;
        impl_->handing_off = false;
        impl_->handoff_kind = Impl::Handoff::none;
        impl_->no_grad_handoff = false;
        impl_->runner.reset();
        impl_->gate_refusals = 0;
        impl_->jg_solve_started = false;
    }

    void NonsmoothNewtonBase::update_solver_info(json &solver_info, const double per_iteration)
    {
        const auto &s = impl_->st;
        const long own = s.path_iters + s.ls_iters + s.path_fail;
        const long total = solver_info.contains("iterations") ? solver_info["iterations"].get<long>() : own;
        solver_info["NonsmoothNewton"] = {
            {"active_iterations", s.active_iters},
            {"path_iterations", s.path_iters},
            {"line_search_iterations", s.ls_iters},
            {"path_failures", s.path_fail},
            {"fallback_iterations", total - own},
            {"failed_directions", s.failed},
            {"inner_steps", s.inner},
            {"factorizations", s.factorizations},
            {"factorization_failures", s.factorization_failures},
            {"back_substitutions", s.solves},
            {"true_energy_evaluations", s.true_energy},
            {"retractions", s.retractions},
            {"projected_iterations", s.projected},
            {"nonsmooth_updates", s.rebuilds},
            {"handoff_noise_floor", s.handoff_noise},
            {"handoff_grad_norm", s.handoff_grad},
            {"handoff_iterations", s.handoff_iters},
            {"hand_backs", s.hand_backs},
            {"jacobian_floor_hits", s.jacobian_floor_hits},
            {"jacobian_bound_hits", s.jacobian_bound_hits}};
        json &info = solver_info["NonsmoothNewton"];
        if (opts_.first_layer_on())
            info["ev_promotions"] = s.promotions;
        if (opts_.reread_on())
            info.update({{"reread_calls", s.rr_calls}, {"reread_after_promotion", s.rr_after_promotion}, {"restart_cont_n", s.restart_cont}});
        if (opts_.selection == "blind")
            info.update({{"ev_select_calls", s.select_calls}, {"ev_selected", s.selected}});
    }
} // namespace polysolve::nonlinear
