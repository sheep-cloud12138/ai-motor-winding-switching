# -*- coding: utf-8 -*-
"""Bisect which build step breaks FEMM's material-assignment check."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401
from exp.femm_validate import Femm, N_SLOT, SLOT_W

RIN, ROUT = 0.005, 0.065
R_SHAFT, R_ROTOR, R_BORE = 0.012, 0.0295, 0.0310
R_SLOT_IN, R_SLOT_OUT = 0.0355, 0.0475

f = Femm()
step = [0]

def begin():
    f.raw('newdocument(0)')
    r = f.mi_probdef(0, "meters", "planar", 0.1)
    f.mi_addmaterial("Air", 1, 1, 0, 0)
    f.mi_addmaterial("IronLin", 4000, 4000, 0, 0)
    for k in range(N_SLOT):
        f.mi_addcircprop(f"c%d" % k, 0, 1)
    f.mi_addboundprop("azero", 0, 0, 0, 0, 0, 0, 0, 0, 0)

def label(x, y, mat, circ=""):
    f.mi_addblocklabel(x, y)
    f.mi_selectlabel(x, y)
    f.mi_setblockprop(mat, 1, 0.001, circ, 0, 0, 0)
    f.mi_clearselected()

def try_solve(tag):
    step[0] += 1
    try:
        f.mi_saveas("C:/Users/19072/AppData/Local/Temp/femmtest/bisect%d.fem" % step[0])
        f.mi_analyze(1)
        f.mi_loadsolution()
        print("step %d (%s): SOLVE OK" % (step[0], tag), flush=True)
        return True
    except Exception as e:
        print("step %d (%s): FAIL %s" % (step[0], tag, str(e)[:60]), flush=True)
        return False

begin()

# Stage 1: full air disk (shaft hole only)
circle_r = [RIN, ROUT]
def circle(r, extra=()):
    angs = sorted({a % 360 for a in list(extra) + [0, 90, 180, 270]})
    P = []
    for a in angs:
        ar = np.deg2rad(a)
        x, y = r*np.cos(ar), r*np.sin(ar)
        f.mi_addnode(x, y); P.append((x, y))
    for i in range(len(P)):
        x1, y1 = P[i]; x2, y2 = P[(i+1) % len(P)]
        da = (angs[(i+1) % len(angs)] - angs[i]) % 360
        f.mi_addarc(x1, y1, x2, y2, da, 3)

for r in (RIN, ROUT):
    circle(r)
label(0.03, 0.01, "Air")
# A=0 on outer boundary
f.mi_selectsegment(ROUT, 0)
f.mi_setsegmentprop("azero", 2, 0, 0, 0)
f.mi_clearselected()
try_solve("air annulus")

# Stage 2: add iron annulus (rotor) inside
circle(R_SHAFT)
label(0.018, 0.02, "IronLin")
try_solve("+ rotor iron")

# Stage 3: PM ring as air annulus
circle(R_ROTOR - 0.004); circle(R_ROTOR)
label(0.0275*np.cos(1.2), 0.0275*np.sin(1.2), "Air")
try_solve("+ PM ring (air)")

# Stage 4: airgap + stator bore circles
circle(R_BORE)
label(0.0302*np.cos(0.5), 0.0302*np.sin(0.5), "Air")
try_solve("+ airgap")

# Stage 5: slot circles with wedge nodes
wedge = []
for k in range(N_SLOT):
    c = 15 + 30*k
    wedge += [c - np.degrees(SLOT_W/2), c + np.degrees(SLOT_W/2)]
circle(R_SLOT_IN, wedge)
circle(R_SLOT_OUT, wedge)
try_solve("+ slot circles (no labels)")

# Stage 6: wedge radial edges
for k in range(N_SLOT):
    c = np.deg2rad(15 + 30*k); h = SLOT_W/2
    for sgn in (-1, 1):
        a = c + sgn*h
        f.mi_addsegment(R_SLOT_IN*np.cos(a), R_SLOT_IN*np.sin(a),
                        R_SLOT_OUT*np.cos(a), R_SLOT_OUT*np.sin(a))
try_solve("+ wedge edges")

# Stage 7: stator iron labels (yoke, tooth-top, one tooth)
label(0.056*np.cos(0.3), 0.056*np.sin(0.3), "IronLin")
label(0.0333*np.cos(0.3), 0.0333*np.sin(0.3), "IronLin")
label(0.0415, 0.0, "IronLin")
try_solve("+ stator iron labels")

# Stage 8: coil label in slot 0
c = np.deg2rad(15)
label(0.0415*np.cos(c), 0.0415*np.sin(c), "Air", circ="c0")
try_solve("+ one coil label")
