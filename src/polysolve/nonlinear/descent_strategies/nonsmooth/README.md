# NonsmoothNewton: modules

`NonsmoothNewton` (and `BlindNewton`, selection = blind) is Newton with a protected path on the split f = s + n of the problem: a smooth
part s whose quadratic model is frozen at the outer iterate, and a nonsmooth part n that the reduced model keeps exact. The files in this
directory are the parts of the outer iteration; `../NonsmoothNewtonBase.{hpp,cpp}` holds the order of the calls, the handoff to the Newton chain
and the hand-back, `../NonsmoothNewton.{hpp,cpp}` and `../BlindNewton.{hpp,cpp}` are the two selections.

| file | what it does |
|---|---|
| `BasisMatrix` | B = H_s + proj(H_n) and its pieces, the factorization (symbolic analysis only when the pattern changes), the direction -B^-1 g, the fallback to proj(H_s) + proj(H_n), in-place updates of B's pieces when units become exact, and the local repair of B (`basis_repair/mode = r_ad`) |
| `SubspaceModel` (`model/reduced_algebra`, `model/row_major_q`) | the reduced model m(z) = ell^T z + 1/2 z^T G z + n(x + Q z) - n(x) over the M-orthonormal columns Q, the re-read correction, the Levenberg-Marquardt step |
| `ReducedMatrix` | A = G + Q^T K Q from the local blocks of n: block by block (`per_block`) or through a sparse structure fixed at the first point of an outer iteration (`merged`, `auto`) |
| `PathRunner` | the inner loop: limiters (guard, validity, fraction to boundary, `max_step_size`, Jacobian floor and bound), the first layer (units behind a cut become exact, the polyline restarts), the re-read of the true gradient, acceptance of the end point or a retraction along the polyline |
| `../../NonsmoothPorts.hpp` | the ports a problem gives the strategy: `FirstLayerPort`, `SelectionPort`, `BasisPort`; `LimitEvent`, `NonsmoothDelta`, the Jg gate |

Method parts (all in the spec under `/NonsmoothNewton`; `auto` is off for `energy_split` and on for `blind`):
`path/always`, `reread/enabled`, `first_layer/*`, `handoff_gate/*`, `jacobian/exact_gate*`, `reduced_matrix/*`, `basis_repair/*`. With the
defaults of `energy_split` the outputs are those of the paper configuration (bitwise).

Tests: `tests/test_nonsmooth_newton.cpp` (options, the path with a log term, re-read, first layer with a mock port, Jg, `ReducedMatrix`
against block-by-block with its integer checks and negative controls, the CHOLMOD failure pivot).
