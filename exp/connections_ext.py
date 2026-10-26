# -*- coding: utf-8 -*-
"""Extended winding connections: 4P (full parallel), tap50 (half-winding tap),
openB (phase-B open circuit, isolated neutral).

All are still linear maps i_c = C i_p on the 12 coil currents, consistent with
the connection-as-constraint formulation used in the paper.
"""

import numpy as np

from gen_coil_dataset import PHASE_COILS, connection_currents as _base


def connection_currents_ext(kind, ip):
    if kind in ("4S", "2S2P", "delta", "asym"):
        return _base(kind, ip)
    ic = np.zeros(12)
    if kind == "4P":
        for p, key in enumerate("ABC"):
            coils, signs = PHASE_COILS[key]
            ic[coils] = 0.25 * signs * ip[p]
    elif kind == "tap50":
        # Series tap: only the first two coils of each phase are in circuit.
        for p, key in enumerate("ABC"):
            coils, signs = PHASE_COILS[key]
            ic[coils[:2]] = signs[:2] * ip[p]
    elif kind == "openB":
        # Phase B open, star point isolated: iA + iC = 0.
        coilsA, sA = PHASE_COILS["A"]
        coilsC, sC = PHASE_COILS["C"]
        ic[coilsA] = sA * ip[0]
        ic[coilsC] = sC * (-ip[0])
    else:
        raise ValueError(kind)
    return ic
