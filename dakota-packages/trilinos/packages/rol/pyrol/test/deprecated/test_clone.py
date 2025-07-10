import numpy as np
from PyROL.vectors import npVector

x = npVector(np.ones(3))
y = npVector(np.ones(3))

x.axpy(2.0, y)
assert np.all(x.values == 3.0)
if hasattr(x, "copies"):
    assert len(x.copies) == 0, x.copies
if hasattr(y, "copies"):
    assert len(y.copies) == 0, y.copies
