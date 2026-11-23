#include "BlindNewton.hpp"

#include <polysolve/nonlinear/NonsmoothPorts.hpp>

namespace polysolve::nonlinear
{
    std::string BlindNewton::name() const
    {
        return "NonsmoothNewton";
    }

    SelectionPort *BlindNewton::selection_port(Problem &problem)
    {
        // a problem without the term (a full-size phase that the problem leaves to the Newton chain) has no selection either
        if (!problem.has_nonsmooth_term())
            return nullptr;
        SelectionPort *const port = problem.nonsmooth_selection();
        if (port == nullptr)
            throw std::runtime_error("NonsmoothNewton: selection = blind needs a problem that gives a selection (NonsmoothTerm::selection)");
        return port;
    }
} // namespace polysolve::nonlinear
