#include "BasisMatrix.hpp"
#include "../ParallelKernels.hpp"

#include <algorithm>

namespace polysolve::nonlinear
{
    BasisMatrix::BasisMatrix(const NonsmoothNewtonBase::Options &opts, NonsmoothNewtonBase::Stats &stats,
                             const json &solver_params, const json &linear_solver_params, spdlog::logger &logger)
        : opts_(opts), st_(stats), logger_(logger)
    {
        res_tol_ = extract_param("Newton", "residual_tolerance", solver_params);
        linear_ = polysolve::linear::Solver::create(linear_solver_params, logger_);
    }

    void BasisMatrix::check_solver()
    {
        // the projected-mode switch relies on the factorization reporting an indefinite B (an LDLT backend would not)
        if (linear_->name() != "Eigen::CholmodSupernodalLLT" && linear_->name() != "Eigen::SimplicialLLT")
            log_and_throw_error(logger_, "NonsmoothNewton: needs an LLT linear solver that reports indefinite matrices (Eigen::CholmodSupernodalLLT or Eigen::SimplicialLLT), got {}", linear_->name());
    }

    /// Symbolic analysis only when the sparsity pattern differs from the last analysis (the numeric factorization of
    /// an unchanged pattern does not depend on redoing it). The analysis is symbolic, so it stays valid after a failed
    /// numeric factorization: the projected retry of an outer iteration usually has the same pattern.
    void BasisMatrix::factorize(const SpMat &H)
    {
        if (!same_pattern(H))
        {
            pattern_ok_ = false;
            linear_->analyze_pattern(H, H.rows());
            if (H.isCompressed())
            {
                pattern_outer_.assign(H.outerIndexPtr(), H.outerIndexPtr() + H.outerSize() + 1);
                pattern_inner_.assign(H.innerIndexPtr(), H.innerIndexPtr() + H.nonZeros());
                pattern_rows_ = H.rows();
                pattern_ok_ = true;
            }
        }
        ++st_.factorizations;
        try
        {
            linear_->factorize(H);
        }
        catch (...)
        {
            ++st_.factorization_failures;
            throw;
        }
    }

    bool BasisMatrix::same_pattern(const SpMat &H) const
    {
        return pattern_ok_ && H.isCompressed() && H.rows() == pattern_rows_ && H.cols() == pattern_rows_
               && size_t(H.nonZeros()) == pattern_inner_.size()
               && std::equal(pattern_outer_.begin(), pattern_outer_.end(), H.outerIndexPtr())
               && std::equal(pattern_inner_.begin(), pattern_inner_.end(), H.innerIndexPtr());
    }

    BasisMatrix::Vec BasisMatrix::backsolve(const Vec &rhs)
    {
        ++st_.solves;
        Vec out = Vec::Zero(rhs.size());
        linear_->solve(rhs, out);
        return out;
    }

    namespace
    {
        /// Subtracts the entries of D from H in place (the entries must be in the pattern of H; entries of D that are exactly
        /// zero need not be). False, with H untouched, if one is missing.
        bool subtract_in_place(StiffnessMatrix &H, const StiffnessMatrix &D)
        {
            using SpMat = StiffnessMatrix;
            if (!H.isCompressed())
                H.makeCompressed();
            std::vector<Eigen::Index> pos;
            pos.reserve(size_t(D.nonZeros()));
            for (Eigen::Index j = 0; j < D.outerSize(); ++j)
            {
                const SpMat::StorageIndex *b = H.innerIndexPtr() + H.outerIndexPtr()[j];
                const SpMat::StorageIndex *e = H.innerIndexPtr() + H.outerIndexPtr()[j + 1];
                for (SpMat::InnerIterator it(D, j); it; ++it)
                {
                    const SpMat::StorageIndex *q = std::lower_bound(b, e, SpMat::StorageIndex(it.row()));
                    if (q == e || *q != it.row())
                    {
                        if (it.value() == 0.0)
                        {
                            pos.push_back(-1);
                            continue;
                        }
                        return false;
                    }
                    pos.push_back(q - H.innerIndexPtr());
                }
            }
            size_t k = 0;
            for (Eigen::Index j = 0; j < D.outerSize(); ++j)
                for (SpMat::InnerIterator it(D, j); it; ++it, ++k)
                    if (pos[k] >= 0)
                        H.valuePtr()[pos[k]] -= it.value();
            return true;
        }
    } // namespace

    bool BasisMatrix::solve_plain(const Vec &rhs, Vec &out)
    {
        out = Vec::Zero(rhs.size());
        linear_->solve(rhs, out);
        return true;
    }

    bool BasisMatrix::apply_delta(const NonsmoothDelta &delta)
    {
        if (delta.n == 0)
            return false;
        if (!subtract_in_place(Hnc, delta.dH))
        {
            SpMat t = Hnc - delta.dH;
            t.makeCompressed();
            Hnc = std::move(t);
        }
        gnc -= delta.dg;
        gc0 += delta.dg;
        Ec0 += delta.dE;
        reanchored = true;
        return true;
    }

    void BasisMatrix::assemble_parts(Problem &p, const Vec &x)
    {
        reanchored = false;
        p.smooth_hessian(x, false, Hnc);
        p.smooth_gradient(x, gnc);
        p.nonsmooth_hessian(x, true, Hcp);
        parallel::spadd(Hnc, Hcp, B);
        if (opts_.first_layer_on() && opts_.lift_model == "frozen")
        {
            // frozen lifted units: B has their projected block (the model keeps the frozen one)
            BasisPort *port = p.nonsmooth_basis();
            SpMat dB;
            if (port != nullptr && port->correction(x, false, dB) && dB.nonZeros() > 0)
            {
                SpMat t = B + dB;
                t.makeCompressed();
                B = std::move(t);
                reanchored = true; // B != Hnc + Hcp
            }
        }
        p.nonsmooth_gradient(x, gc0);
        Ec0 = p.nonsmooth_value(x);
    }

    bool BasisMatrix::try_solve(Problem &p, const SpMat &H, const Vec &g, Vec &d, const NormType nt)
    {
        try
        {
            factorize(H);
        }
        catch (const std::runtime_error &)
        {
            return false;
        }
        d = backsolve(-g);
        Vec Hd(H.rows());
        parallel::spmv(H, d, Hd);
        const double res = p.grad_norm(Vec(Hd + g), nt);
        return !(std::isnan(res) || res > res_tol_);
    }

    bool BasisMatrix::direction_projected_mode(Problem &p, const Vec &x, const Vec &g, Vec &d, const double h_edge)
    {
        projected = false;
        bool ok = try_solve(p, B, g, d, NormType::EUCLIDEAN);
        if (ok && opts_.guard)
        {
            const double len = p.nonsmooth_step_length(x, x + d);
            if (len > opts_.guard_check * h_edge && p.smooth_valid_fraction(x, x, x + d) < 0.25)
                ok = false;
        }
        if (ok)
            return true;
        {
            SpMat h;
            p.smooth_hessian(x, true, h);
            parallel::spadd(h, Hcp, B);
            Hnc = std::move(h); // projected smooth quadratic in the model (bounded below)
        }
        projected = true;
        ++st_.projected;
        return try_solve(p, B, g, d, NormType::EUCLIDEAN);
    }
} // namespace polysolve::nonlinear
