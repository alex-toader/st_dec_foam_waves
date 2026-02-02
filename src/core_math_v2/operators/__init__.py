"""DEC operators - incidence matrices, Hodge Laplacian, cycle space, parity."""

from .incidence import (
    build_d0,
    build_d1,
    build_incidence_matrices,
    build_hodge_laplacian,
    get_cycle_space,
    build_operators_from_mesh,
)

from .parity import (
    build_parity_operator,
    parity_decomposition,
    build_parity_from_mesh,
)
