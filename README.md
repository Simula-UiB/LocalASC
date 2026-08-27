# LocASC
Local ASC repo

This repository is organized as follows:

'''/Codes''' contains parity check matrices of the tested codes, each in its corresponding file with notation '''n_k_d_'''.

''' /examples ''' contains compiler runs to obtain circuits and their corresponding logical error rate experiments and circuit distance calculation.

'''/circuit_builder''' contains the files needed to build stabilizer measurement circuits from PCMs with respect to a schedule given by different methods (Auto-Stabilizer check [\addref] ATB for Coset qLDPC codes [\addref] ). Also contains the functions needed to add noise (Phenomenological noise, Circuit-Level noise)

