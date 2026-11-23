#pragma once

#include "NonsmoothNewtonBase.hpp"

namespace polysolve::nonlinear
{
    /// @brief NonsmoothNewton with the blind selection (NonsmoothNewton/selection = "blind"): the problem gives no split into
    /// smooth and exact units; the units that are modelled exactly are chosen by the selection port of the problem (second and
    /// third layer at the start of an outer iteration), and by the first layer in the path. See NonsmoothNewtonBase.
    class BlindNewton : public NonsmoothNewtonBase
    {
    public:
        using NonsmoothNewtonBase::NonsmoothNewtonBase;

        std::string name() const override;

    protected:
        SelectionPort *selection_port(Problem &problem) override;
    };
} // namespace polysolve::nonlinear
