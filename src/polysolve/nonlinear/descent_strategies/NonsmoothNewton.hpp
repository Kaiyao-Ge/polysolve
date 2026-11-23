#pragma once

#include "NonsmoothNewtonBase.hpp"

namespace polysolve::nonlinear
{
    /// @brief NonsmoothNewton with the split f = s + n that the problem gives (NonsmoothNewton/selection =
    /// "energy_split"): the nonsmooth term n is the part of the energy that is modelled exactly. See
    /// NonsmoothNewtonBase.
    class NonsmoothNewton : public NonsmoothNewtonBase
    {
    public:
        using Options = NonsmoothNewtonBase::Options;

        using NonsmoothNewtonBase::NonsmoothNewtonBase;

        std::string name() const override;
    };
} // namespace polysolve::nonlinear
