python - <<'PY'
from xtb.ase.calculator import XTB
from ase import Atoms

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
atoms.calc = XTB(method="GFN2-xTB")

print("energy:", atoms.get_potential_energy())
print("forces:", atoms.get_forces())
print("xTB ASE interface OK")
PY

# python - <<'PY'
# import xtb, sys
# print("xtb file:", xtb.__file__)
# print("sys.path first entries:")
# print("\n".join(sys.path[:10]))
# PY