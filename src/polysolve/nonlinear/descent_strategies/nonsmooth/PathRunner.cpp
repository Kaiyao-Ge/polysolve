#include "PathRunner.hpp"
#include "../ParallelKernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace polysolve::nonlinear
{
    namespace
    {
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
        constexpr double EPS = std::numeric_limits<double>::epsilon();

        /// One segment of the path (nonsmooth_segment_begin / end).
        struct SegmentScope
        {
            Problem &p;
            bool active = false;
            explicit SegmentScope(Problem &p_) : p(p_) {}
            void begin(const Eigen::VectorXd &a, const Eigen::VectorXd &b)
            {
                p.nonsmooth_segment_begin(a, b);
                active = true;
            }
            void end()
            {
                if (active)
                    p.nonsmooth_segment_end();
                active = false;
            }
            ~SegmentScope() { end(); }
        };
    } // namespace

    void PathRunner::reset()
    {
        j_ref = NaN;
        min_j_x = NaN;
        j_floor_missing = j_bound_missing = false;
    }

    double PathRunner::mnorm2(const Vec &v) const
    {
        return std::sqrt((Mred.array() * v.array() * v.array()).sum());
    }

    /// A Jacobian limit is on but the problem gives NaN for it: reported once per solve.
    void PathRunner::jacobian_missing(bool &reported, const char *limit, const char *callback)
    {
        if (reported)
            return;
        reported = true;
        logger_.debug("[NonsmoothNewton] the Jacobian {} is not applied where Problem::{} gives NaN (reported once per solve)", limit, callback);
    }

    /// The Jacobian limits at the end point y of an inner step from the outer iterate x: 0 if both hold, 1 if y is
    /// below the floor jfloor (-inf: off), 2 if an element's volume ratio to x is below the bound.
    int PathRunner::jacobian_violation(Problem &p, const Vec &x, const Vec &y, const double jfloor)
    {
        double min_j = NaN, min_ratio = NaN; // both from one sweep over the elements
        p.smooth_min_jacobians(x, y, jfloor > -INFINITY, opts_.jacobian_bound > 0, min_j, min_ratio);
        if (jfloor > -INFINITY && !(min_j >= jfloor))
            return 1;
        if (opts_.jacobian_bound > 0)
        {
            if (!std::isfinite(min_ratio))
                jacobian_missing(j_bound_missing, "bound", "smooth_min_jacobians");
            else if (min_ratio < opts_.jacobian_bound)
                return 2;
        }
        return 0;
    }

    /// Nonsmooth term at reduced y; rebuilds its state unless reuse matches y.
    NonsmoothEval PathRunner::eval(Problem &p, const Vec &y, bool deriv, NonsmoothEval *reuse)
    {
        NonsmoothEval c;
        c.y = y;
        if (reuse && reuse->y.size() == c.y.size() && reuse->y == c.y)
        {
            c.E = reuse->E;
            if (reuse->deriv)
            {
                c.deriv = true;
                c.g = reuse->g;
                c.K = reuse->K;
                c.blocks = reuse->blocks;
                c.local = reuse->local;
                return c;
            }
        }
        else
        {
            p.nonsmooth_update(c.y);
            last_built_ = c.y;
            ++st_.rebuilds;
            c.E = p.nonsmooth_value(c.y);
        }
        if (deriv)
        {
            c.blocks.swap(spare_blocks_); // the allocation of a replaced state
            c.blocks.clear();
            c.local = p.nonsmooth_hessian_blocks(c.y, c.g, c.blocks);
            if (!c.local)
            {
                p.nonsmooth_gradient(c.y, c.g);
                p.nonsmooth_hessian(c.y, false, c.K);
            }
            c.deriv = true;
        }
        return c;
    }

    double PathRunner::true_energy(Problem &p, const Vec &y)
    {
        ++st_.true_energy;
        if (opts_.dedupe)
        {
            // nonsmooth state already rebuilt at y (inside the segment): the objective there without another rebuild
            if (last_built_.size() == y.size() && last_built_ == y)
                return p.value(y);
        }
        p.solution_changed(y);
        last_built_.resize(0);
        return p.value(y);
    }

    // ------------------------------------------------------------------------------------------------ protected path
    bool PathRunner::run(Problem &p, const Vec &x, const Vec &g, double E0, const Vec &pN, Vec &step, int &n_inner, const PathContext &ctx)
    {
        last_built_.resize(0); // the nonsmooth state is only tracked inside this function
        noise_stop = false;
        info = PathInfo();
        FirstLayerPort *const fl = ctx.first_layer;
        SelectionPort *const sel = ctx.selection;
        JGate *const jg = ctx.jgate && sel != nullptr ? &sel->jgate() : nullptr;
        bool jg_seen = false; // a limit event of an exact unit that is not enforced was already counted in this path
        bool seg1_cont = false; // segment 1 of the polyline (the start to the point of the last restart) carries the offset of the cumulative value
        const int inner_max = opts_.max_inner;
        const int n = int(x.size());
        const int cap = std::min(opts_.max_columns, inner_max) + 2;
        // smallest change of the true energy the acceptance test can resolve (0: model rounding only)
        const double e_res = opts_.resolution ? 32 * EPS * std::abs(E0) : 0.0;
        const int col_cap = opts_.max_columns;
        SubspaceModel M(basis_, reduced_, n, cap);
        // Jacobian floor of this outer iteration (-inf: off): jacobian_floor * J_ref, or min J at x where x is below it
        double jfloor = -INFINITY;
        if (opts_.jacobian_floor > 0 && j_ref > 0 && std::isfinite(min_j_x))
            jfloor = min_j_x >= opts_.jacobian_floor * j_ref ? opts_.jacobian_floor * j_ref : min_j_x;
        bool j_end = false; // a Jacobian limit shortened the first step: the path ends with it

        const double npn = mnorm2(pN);
        double seg_cap = INFINITY;
        if (opts_.guard)
        {
            const double pN_len = p.nonsmooth_step_length(x, x + pN);
            double alpha_v = 1;
            if (pN_len > opts_.guard_check * h_edge)
                alpha_v = p.smooth_valid_fraction(x, x, x + pN);
            seg_cap = std::min(opts_.guard_c * alpha_v * pN_len, opts_.guard_abs * h_edge) + p.nonsmooth_activation_distance();
        }
        if (!(npn > 0) || !std::isfinite(npn))
        {
            return false;
        }
        M.add_column(pN / npn);
        M.rr = ctx.reread;

        Vec z = Vec::Zero(cap);
        // z = 0 evaluation: the nonsmooth state is at x already
        NonsmoothEval cstate;
        cstate.y = x;
        cstate.E = basis_.Ec0;
        cstate = eval(p, x, true, &cstate); // fills derivatives only (reuse of E)

        std::vector<Vec> pts;      // p^(j) reduced displacements
        std::vector<double> mvals; // m at p^(j)
        std::vector<Vec> zs;
        pts.push_back(Vec::Zero(n));
        mvals.push_back(0.0);
        zs.push_back(Vec::Zero(cap));
        SubspaceModel::Eval me = M.eval(z, cstate, true);
        double m_cur = 0;
        double pred_total = 0;
        Vec last_r;
        double ell_inf = M.ell.head(M.k).cwiseAbs().maxCoeff();
        Vec dist, rate;
        // A product with the basis is reused only where the expression it replaces has the same operands: Q v (Eigen:
        // zero, then += Q v) and x + Q v (Eigen: x, then += Q v) round differently.
        Vec p_last;      // Q z at the last accepted point
        int k_last = -1; // its number of columns

        // ---- re-read: the true total gradient at the current point of the path; cR = g_true - (model gradient), cq = Q^T cR ----
        Vec rr_cR;                // correction of the model gradient of the running path (size 0: zero)
        Vec rr_gcur;              // true total gradient at the current point (x: the gradient the solver gave)
        std::vector<Vec> seg_cq;  // per polyline segment j >= 1: the correction in force when it was taken (retraction points of a segment are evaluated with it); index 0 unused
        std::vector<double> seg_off;
        if (M.rr)
        {
            rr_gcur = g;
            seg_cq.push_back(M.cq);
            seg_off.push_back(0.0);
        }
        // the problem's gradient at y; the state at y is the one the trial evaluation built (last_built_ == y), else it is rebuilt first
        const auto rr_read = [&](const Vec &y) -> bool {
            if (!(last_built_.size() == y.size() && last_built_ == y))
            {
                p.solution_changed(y); // the full state, as the true energy at y would build it
                last_built_ = y;
            }
            p.gradient(y, rr_gcur);
            return rr_gcur.size() == y.size() && rr_gcur.allFinite();
        };
        // cR, cq and m_off at the current point pk (state cstate, true gradient rr_gcur, model value m_cur): the model gradient there
        // becomes the true one, the value stays
        const auto rr_set = [&](const Vec &pk, const bool after_promotion) {
            const int k = M.k;
            Vec hp(n);
            parallel::spmv(basis_.Hnc, pk, hp);
            Vec mg = basis_.gnc + hp;
            mg += cstate.g;
            rr_cR = rr_gcur - mg;
            M.cq.setZero();
            M.cq.head(k) = M.Q.leftCols(k).transpose() * rr_cR;
            M.m_off = 0;
            const SubspaceModel::Eval e0 = M.eval(z, cstate, false);
            M.m_off = m_cur - e0.m;
            ++(after_promotion ? st_.rr_after_promotion : st_.rr_calls);
        };

        int rounds = 0;
        bool promote = fl != nullptr; // units may still be promoted in this path

        // The units that became exact are taken into the model in place and the path goes on from the current point p_cur with
        // the new model (the polyline restarts there). False if nothing was waiting (the caller then cuts as without the layer).
        const auto restart_after_promotion = [&](const Vec &p_cur) -> bool {
            const double m_before = m_cur; // the cumulative value before the restart
            NonsmoothDelta D;
            if (!fl->take_delta(x, basis_.projected, D) || D.n == 0)
                return false;
            basis_.apply_delta(D);
            M.apply_delta(D);
            if (++rounds >= opts_.lift_max_rounds)
                promote = false; // the last permitted round: later limits only cut
            spare_blocks_.swap(cstate.blocks);
            cstate = eval(p, x + p_cur, true);
            if (M.rr)
            {
                // the old correction contains the force errors of the promoted units: the model without it first, then the correction of the changed model
                rr_cR.resize(0);
                M.cq.setZero();
                M.m_off = 0;
            }
            me = M.eval(z, cstate, true);
            m_cur = me.m;
            const double m_raw_after = m_cur;
            // the cumulative model value survives the change of the model (the offset set in rr_set below, and in the segment of the
            // pre-restart point, makes the model value at the current point equal to its value before the restart)
            const bool cont = opts_.restart_continuity && M.rr;
            if (cont)
            {
                m_cur = m_before;
                ++st_.restart_cont;
            }
            pred_total = -m_cur;
            ell_inf = M.ell.head(M.k).cwiseAbs().maxCoeff();
            const bool moved = pts.size() > 1;
            pts.clear();
            mvals.clear();
            zs.clear();
            pts.push_back(Vec::Zero(n));
            mvals.push_back(0.0);
            zs.push_back(Vec::Zero(cap));
            if (M.rr)
            {
                seg_cq.clear();
                seg_off.clear();
                seg_cq.push_back(M.cq);
                seg_off.push_back(0.0);
            }
            if (moved)
            {
                pts.push_back(p_cur);
                mvals.push_back(m_cur);
                zs.push_back(z);
                seg1_cont = cont;
                if (M.rr)
                {
                    seg_cq.push_back(M.cq); // the segment from the start to the current point: the changed model without the correction
                    seg_off.push_back(cont ? M.m_off + (m_before - m_raw_after) : M.m_off);
                }
            }
            if (M.rr)
            {
                rr_set(p_cur, true); // the true gradient at the current point is known (rr_gcur): the change was in the model only
                me = M.eval(z, cstate, true);
            }
            return true;
        };

        for (int it = 0; it < inner_max; ++it)
        {
            if (me.grad.cwiseAbs().maxCoeff() <= 1e-10 * std::max(1.0, ell_inf))
            {
                break; // small reduced gradient
            }
            Vec dz;
            if (!SubspaceModel::lm_step(me.A, me.grad, dz))
            {
                break; // LM failure
            }
            int k = M.k;
            Vec zk = z.head(k);
            const Vec p_cur = k == k_last ? p_last : Vec(M.Q.leftCols(k) * zk);
            Vec pe, xe; // Q (zk + dz) and x + Q (zk + dz), while dz is unchanged (pe_ok, xe_ok)
            bool pe_ok = false, xe_ok = false;
            if (opts_.guard)
            {
                pe = M.Q.leftCols(k) * (zk + dz);
                pe_ok = true;
                double len = p.nonsmooth_step_length(x + p_cur, x + pe);
                if (len > seg_cap)
                {
                    dz *= seg_cap / len; // scale, never reject
                    len = seg_cap;
                    pe_ok = false;
                    ++info.trunc[0];
                }
                // with fast_validity every segment end is checked for validity; otherwise only
                // segments longer than guard_check * h
                if (len > opts_.guard_check * h_edge || opts_.fast_validity)
                {
                    xe = x + M.Q.leftCols(k) * (zk + dz);
                    xe_ok = true;
                    const double a = p.smooth_valid_fraction(x, x + p_cur, xe);
                    if (a < 1)
                    {
                        dz *= a;
                        pe_ok = xe_ok = false;
                        ++info.trunc[1];
                    }
                }
                if (!(dz.cwiseAbs().maxCoeff() > 0))
                {
                    break; // guard reduced the step to zero
                }
            }
            if (opts_.ftb_tau > 0)
            {
                // fraction-to-boundary on the terms active at the segment start (linearized distance)
                if (!xe_ok)
                {
                    xe = x + M.Q.leftCols(k) * (zk + dz);
                    xe_ok = true;
                }
                p.nonsmooth_active_distances(x + p_cur, xe, dist, rate);
                double sfac = 1;
                for (int i = 0; i < dist.size(); ++i)
                {
                    if (rate[i] < 0)
                    {
                        const double sc = (1 - opts_.ftb_tau) * dist[i] / -rate[i];
                        sfac = std::min(sfac, sc);
                    }
                }
                if (sfac < 1)
                {
                    ++info.trunc[2];
                    if (promote && (ctx.events >> 2 & 1))
                    {
                        const Vec xa = x + p_cur;
                        LimitEvent ev{LimitEvent::Kind::ftb, x, n_inner};
                        ev.a = &xa;
                        ev.b = &xe;
                        ev.ftb_tau = opts_.ftb_tau;
                        const int nv = fl->identify(ev);
                        st_.promotions += nv;
                        if (nv > 0 && restart_after_promotion(p_cur))
                        {
                            --it;
                            continue;
                        }
                    }
                    dz *= sfac;
                    pe_ok = xe_ok = false;
                }
            }
            const Vec p_end = pe_ok ? pe : Vec(M.Q.leftCols(k) * (zk + dz));
            double beta;
            SegmentScope seg(p);
            // one candidate set for the whole segment
            seg.begin(x + p_cur, Vec(x + p_end));
            beta = p.max_step_size(x + p_cur, x + p_end);
            if (!(beta > 0))
            {
                break; // CCD step zero
            }
            double alpha = beta >= 1 ? 1.0 : 0.99 * beta;
            if (beta < 1)
            {
                ++info.trunc[3];
                if (promote && (ctx.events >> 3 & 1))
                {
                    const Vec xa = x + p_cur, xb = x + p_end;
                    LimitEvent ev{LimitEvent::Kind::ccd, x, n_inner};
                    ev.a = &xa;
                    ev.b = &xb;
                    const int nv = fl->identify(ev);
                    st_.promotions += nv;
                    if (nv > 0)
                    {
                        seg.end();
                        if (restart_after_promotion(p_cur))
                        {
                            --it;
                            continue;
                        }
                    }
                }
            }
            const double slope = me.grad.dot(dz);
            bool found = false;
            NonsmoothEval ctrial;
            SubspaceModel::Eval mtrial;
            Vec ztrial;
            for (int bt = 0; bt < 50; ++bt, alpha *= 0.5)
            {
                ztrial = z;
                ztrial.head(k) = zk + alpha * dz;
                // zk + 1 dz is zk + dz exactly
                const Vec y = alpha == 1.0 && xe_ok ? xe : Vec(x + M.Q.leftCols(k) * ztrial.head(k));
                ctrial = eval(p, y, false);
                if (!std::isfinite(ctrial.E))
                    continue;
                mtrial = M.eval(ztrial, ctrial, false);
                if (mtrial.m <= m_cur + 1e-4 * alpha * slope + 32 * EPS * (mtrial.sumabs + me.sumabs))
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                break; // Armijo failure
            }
            // Jacobian limits at the end point: a later step that violates one is not taken (the path ends at the
            // previous point); a first step is halved further until the limits and the Armijo condition hold, and
            // the path ends after it
            int jv = jfloor > -INFINITY || opts_.jacobian_bound > 0 ? jacobian_violation(p, x, ctrial.y, jfloor) : 0;
            if (jv == 0 && jg != nullptr && !jg->enabled && opts_.jacobian_bound > 0)
            {
                // Jg: the limit is off for the exact units; the first violating step of every `retest`-th outer iteration enforces it
                const double mr = sel->min_ratio_exact(x, ctrial.y);
                if (std::isfinite(mr) && mr < opts_.jacobian_bound)
                {
                    info.jg_event = true;
                    if (!jg_seen)
                    {
                        jg_seen = true;
                        ++jg->dis_events;
                        if (jg->dis_events % jg->retest == 0)
                        {
                            sel->jratio_exact(1);
                            jg->applied = 1;
                            jv = 2;
                        }
                    }
                }
            }
            if (jv != 0)
            {
                if (fl != nullptr && (ctx.events >> (jv == 1 ? 4 : 5) & 1))
                {
                    LimitEvent ev{jv == 1 ? LimitEvent::Kind::jacobian_floor : LimitEvent::Kind::jacobian_bound, x, n_inner};
                    ev.a = &x;
                    ev.b = &ctrial.y;
                    ev.jacobian_floor = jfloor;
                    ev.jacobian_bound = opts_.jacobian_bound;
                    ev.promote = promote; // without rounds left the units are only recorded for the next outer iteration
                    const int nv = fl->identify(ev);
                    st_.promotions += nv;
                    if (promote && nv > 0)
                    {
                        seg.end();
                        if (restart_after_promotion(p_cur))
                        {
                            --it; // the promotion does not use up an inner step
                            continue;
                        }
                    }
                }
                ++(jv == 1 ? st_.jacobian_floor_hits : st_.jacobian_bound_hits);
                ++info.trunc[jv == 1 ? 4 : 5];
                if (jg != nullptr && jv != 1 && jg->applied == 1)
                    info.jg_end = info.jg_event = true; // the limit of an exact unit ends the path here
                if (pts.size() > 1)
                {
                    break;
                }
                bool held = false;
                for (int h = 0; h < 40; ++h)
                {
                    alpha *= 0.5;
                    ztrial = z;
                    ztrial.head(k) = zk + alpha * dz;
                    const Vec y = x + M.Q.leftCols(k) * ztrial.head(k);
                    ctrial = eval(p, y, false);
                    if (!std::isfinite(ctrial.E) || jacobian_violation(p, x, y, jfloor) != 0)
                        continue;
                    mtrial = M.eval(ztrial, ctrial, false);
                    if (mtrial.m <= m_cur + 1e-4 * alpha * slope + 32 * EPS * (mtrial.sumabs + me.sumabs))
                    {
                        held = true;
                        break;
                    }
                }
                if (!held)
                {
                    break;
                }
                j_end = true;
            }
            const double dec = m_cur - mtrial.m;
            if (dec <= 32 * EPS * (mtrial.sumabs + me.sumabs) + e_res)
            {
                noise_stop = true;
                break; // decrease below the noise floor (step not accepted)
            }
            z = ztrial;
            m_cur = mtrial.m;
            // a full step (alpha 1: not halved by the line search or a Jacobian limit) ends at p_end
            const Vec p_new = alpha == 1.0 ? p_end : Vec(M.Q.leftCols(k) * z.head(k));
            p_last = p_new;
            k_last = k;
            pts.push_back(p_new);
            mvals.push_back(m_cur);
            zs.push_back(z);
            if (M.rr)
            {
                seg_cq.push_back(M.cq); // the correction this step was taken with
                seg_off.push_back(M.m_off);
            }
            ++n_inner;
            const double prev_total = pred_total;
            pred_total = -m_cur;
            const bool relative = opts_.relative_stop > 0 && prev_total > 0 && (pred_total - prev_total) < opts_.relative_stop * pred_total;
            if (M.rr && !j_end && relative)
            {
                // with the re-read the last step needs neither the derivatives at its end nor a new column
                seg.end();
                break;
            }

            // derivatives at the accepted point (reuses the state of the last trial); the blocks of the replaced
            // state are reused
            spare_blocks_.swap(cstate.blocks);
            cstate = eval(p, ctrial.y, true, &ctrial);
            seg.end();
            if (j_end)
                break; // the first step shortened by a Jacobian limit ends the path

            if (M.rr && it + 1 >= inner_max)
                break; // the last permitted step is not read
            if (M.rr)
            {
                if (!rr_read(ctrial.y))
                {
                    return false;
                }
                rr_set(p_new, false);
            }

            // ---- new column from the model residual r = grad m(p) - g - B p ----
            if (M.k < col_cap)
            {
                Vec r;
                if (M.rr && rr_cR.size() == size_t(n))
                {
                    // the true residual directly (the model residual plus cR cancels on the large barrier terms)
                    Vec bp(n);
                    parallel::spmv(basis_.B, p_new, bp);
                    r = rr_gcur - g;
                    r -= bp;
                }
                else if (basis_.projected || basis_.reanchored)
                {
                    Vec hp(n), bp(n);
                    parallel::spmv(basis_.Hnc, p_new, hp);
                    parallel::spmv(basis_.B, p_new, bp);
                    r = hp - bp + cstate.g - basis_.gc0;
                }
                else
                {
                    // as Eigen evaluates r = (g - g0) - H p: r = g - g0, then r -= H p
                    r = cstate.g - basis_.gc0;
                    parallel::spmv_add(basis_.Hcp, p_new, -1.0, r);
                }
                if (r.allFinite() && r.cwiseAbs().maxCoeff() > 0 && !(last_r.size() == r.size() && r == last_r))
                {
                    last_r = r;
                    // M-orthogonalize u = B^-1 r against Q (two passes) and add it unless nearly dependent
                    const Vec u = basis_.backsolve(r);
                    if (u.allFinite())
                    {
                        const double nu = mnorm2(u);
                        Vec v = u;
                        for (int pass = 0; pass < 2; ++pass)
                        {
                            const Vec w = M.Q.leftCols(M.k).transpose() * (Mred.asDiagonal() * v);
                            v -= M.Q.leftCols(M.k) * w;
                        }
                        const double nv = mnorm2(v);
                        if (nu > 0 && nv / nu >= 1e-4 && M.k < cap)
                        {
                            M.add_column(v / nv);
                            // the correction of the model gradient reaches the new column too
                            if (M.rr && rr_cR.size() == size_t(n))
                                M.cq[M.k - 1] = M.Q.col(M.k - 1).dot(rr_cR);
                        }
                    }
                }
            }

            me = M.eval(z, cstate, true);
            if (!M.rr && relative)
            {
                break; // relative model decrease below eta
            }
        }
        if (pts.size() < 2)
            return false;

        // ---- acceptance with polyline retraction ----
        auto try_point = [&](const Vec &pt, double mval) {
            const double pred = -mval;
            if (!(pred > 0))
                return false;
            const double Ey = true_energy(p, x + pt);
            const bool ok = std::isfinite(Ey) && (E0 - Ey) / pred >= opts_.rho_min;
            if (ok)
            {
                info.E_acc = Ey;
            }
            return ok;
        };
        // the model correction of the polyline segment sg (the points of a segment are evaluated with it)
        const auto seg_state = [&](const int sg) {
            if (M.rr && sg < int(seg_cq.size()))
            {
                M.cq = seg_cq[sg];
                M.m_off = seg_off[sg];
            }
        };
        const int K = int(pts.size()) - 1;
        // the model decreases along the polyline, so no retraction point could be verified either
        if (opts_.resolution && !(-mvals[K] > e_res))
        {
            noise_stop = true;
            return false;
        }
        for (int sg = K; sg >= 1; --sg)
        {
            if (sg == K)
            {
                if (try_point(pts[K], mvals[K]))
                {
                    step = pts[K];
                    return true;
                }
            }
            const Vec &pa = pts[sg - 1];
            const Vec &pb = pts[sg];
            if (sg > 1)
            {
                for (double gam : {0.5, 0.25, 0.125})
                {
                    ++st_.retractions;
                    const Vec pt = pa + gam * (pb - pa);
                    Vec zt = zs[sg - 1] + gam * (zs[sg] - zs[sg - 1]);
                    const NonsmoothEval c = eval(p, x + pt, false);
                    seg_state(sg);
                    const SubspaceModel::Eval mm = M.eval(zt, c, false);
                    if (try_point(pt, mm.m))
                    {
                        step = pt;
                        return true;
                    }
                }
            }
            else
            {
                double gam = 0.5;
                for (int h = 0; h < 60 && gam > 1.4e-14; ++h, gam *= 0.5)
                {
                    ++st_.retractions;
                    const Vec pt = gam * pb;
                    Vec zt = gam * zs[1];
                    const NonsmoothEval c = eval(p, x + pt, false);
                    seg_state(1);
                    if (seg1_cont)
                        M.m_off *= gam; // the offset of the cumulative value is interpolated along the segment from the start (value 0) to the restart point
                    const SubspaceModel::Eval mm = M.eval(zt, c, false);
                    if (try_point(pt, mm.m))
                    {
                        step = pt;
                        return true;
                    }
                }
            }
        }
        return false;
    }
} // namespace polysolve::nonlinear
