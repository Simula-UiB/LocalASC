Parity-check matrices (PCMs) for Table I of arXiv:2609.40319v1,
"Local Automorphism-Aware Syndrome Compilation for General Quantum LDPC Codes".

Files
  NN_n<n>_k<k>[_d<d>]_Hx.txt and NN_n<n>_k<k>[_d<d>]_Hz.txt, where NN is the row of Table I
  (01-18, in table order). The distance is omitted from the file name when Table I gives only a
  lower bound. Each file is a whitespace-separated 0/1 matrix with one check per line and one data
  qubit per column. Hx and Hz of the same code use the same column (qubit) order.

These are the fixed pairs of measured CSS PCMs used for Table I. Every PCM except row 18
contains linearly dependent checks. Keep every row: the minimum depth lambda*_SEC is a property
of the measured PCMs, not only of the code they generate.

manifest.csv, one line per row of Table I:
  [[n,k,d]], section, references   as printed in Table I (reference numbers of arXiv:2609.40319v1)
  Hx_file, Hz_file                 the two matrix files
  n, k                             block length and number of logical qubits
  rows_Hx, rows_Hz                 number of rows (checks) of Hx and Hz, dependent rows included
  w                                maximum check weight
  Delta_D                          maximum data-qubit degree (X- and Z-checks together)
  lambda*_SEC                      minimum depth reported in Table I, equal to
                                   w = Delta(G) = max(w, Delta_D)
  |H*|                             order of the automorphism subgroup used in Table I
