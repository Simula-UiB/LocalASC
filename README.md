# Local Automorphism-Aware Syndrome Compilation

This repository contains the official implementation of **Local Automorphism-Aware Syndrome Compilation**. It provides a fully automated pipeline for schedule generation, quantum error correction (QEC) circuit building, and memory experiment evaluation using `stim` and `sinter`.

## Repository Structure

- **/execution**: Command-line interface (CLI) scripts to run the core experiments, filtering pipelines, and table generation.
- **/src**: The core Python library (`LocalASC`), containing the scheduling algorithms, circuit builder, and simulation routing.
- **/codes**: Parity check matrices and code definitions.
- **/paper_results**: Raw logical-error counts behind the figures in the paper.
- **/results**: The default output directory for generated schedules (`.csv`), Stim circuits (`.stim`), Sinter decoding data (`.json`), and generated plots. It is created when you run the scripts and is not tracked by git.
- **/notebooks**: Jupyter notebooks for interactive, step-by-step methodology.

## Installation

Clone the repository and install the required dependencies. The execution scripts and notebooks import the library directly from `src/`, so the package itself does not need to be installed.

```bash
git clone https://github.com/Simula-UiB/LocalASC.git
cd LocalASC
pip install -r requirements.txt
```

## Interactive Notebooks

The `/notebooks` folder contains tutorials for hands-on exploration of the library's capabilities:

* **`circuit_builder_walkthrough.ipynb`**: Demonstrates the step-by-step process of loading binary CSS parity-check matrices, generating an optimal CNOT schedule using the CP-SAT solver, and constructing X- and Z-basis Stim memory circuits with circuit-level noise models.
* **`simulation_tutorial.ipynb`**: Provides a complete QEC simulation workflow. It covers running a decoding experiment using `sinter`, aggregating the results, computing pseudo-thresholds, and utilizing the project's plotting tools to visualize logical error rate (LER) curves.

## Execution Scripts

The `/execution` folder contains the primary entry points for reproducing the results from the paper.

### 1. Circuit Filtering Pipeline

**Script:** `execution/filtering_run.py`

This script automates the search for the optimal parity check schedule. It explores multiple candidate schedules at a specified circuit depth, builds the corresponding Stim memory circuits, and evaluates their logical error rate (LER) using a fast Sinter experiment. The best-performing circuit is automatically saved to `/results/best_circuits`. The default decoder is the sinter BP+OSD-CS10 with Min-Sum, from the `stimbposd` package. This can be customized in `/src/LocalASC/simulation/sinter_experiment`.

**Usage:**
```bash
python execution/filtering_run.py <hx_path> <hz_path> [OPTIONS]
```

**Key Arguments:**
- `--rounds`: Number of syndrome extraction rounds (code distance is suggested).
- `-D, --target_depth`: Target circuit depth for the generated schedules (default: Tanner Graph max_weight).
- `-s, --steps`: Number of candidate schedules to generate and filter (default: 20).
- `-t, --max_time_hrs`: Maximum pipeline runtime in hours (default: 2.0).
- `-p, --eval_p`: Physical error rate used during the fast evaluation phase (default: 0.001).
- `--shots`: Maximum shots per evaluation (default: 10000).
- `--max_errors`: Target errors per evaluation (default: 50).
- `--plot`: Plot the agglomerated results after filtering completes.
- `--save_plot`: Save the plot as a JPG to `results/plots/filtering/`.

### 2. Sinter Memory Experiment

**Script:** `execution/decoding_experiment.py`

Runs a full memory experiment over a range of physical error probabilities. It decodes the provided (or dynamically generated) Stim circuits using BP+OSD and saves the statistical results as a JSON file in `/results/sinter_data/`.

**Usage:**
```bash
python execution/decoding_experiment.py <hx_path> <hz_path> <rounds> -p <prob1> <prob2> ... [OPTIONS]
```

**Key Arguments:**
- `-schedule`: Path to a specific schedule CSV to build circuits from.
- `-D`: Alternatively, provide a target depth to automatically generate a schedule on the fly.
- `--shots`: Maximum number of shots per noise level.
- `--max_errors`: Target number of errors to collect before moving to the next noise level.
- `--plot`: Generate and save a sub-threshold LER plot upon completion.

### 3. Runtime Table Generation (Table I)

**Script:** `execution/runtime_table.py`

Reproduces **Table I** from the paper. This script iterates over standard codes provided in the `/codes` directory, runs the schedule generation algorithm under various policies, and profiles the execution time. 

**Usage:**
```bash
python execution/runtime_table.py [specific_codes ...] [OPTIONS]
```

**Key Arguments:**
- `specific_codes`: Optional space-separated list of specific code names to run (e.g., `04_n36_k4_d4 14_n180_k26_d6`). If omitted, all predefined codes in the suite will be evaluated.
- `--policy`: The subgroup search policy to apply. Choices are `hybrid`, `ladder`, `maxodd`, `odd`, or `full` (default: `hybrid`, the policy used for Table I).
- `--codes_dir`: Path to the directory containing the code matrices (default: `codes/Table1_PCMs`).
- `--output`: Path to save the generated JSONL runtime table (default: `results/timing_tables/localasc_table_timing.jsonl`).
- `--runs`: Number of profiling iterations to run per code (default: 10).
- `--tlim`: Maximum time limit in seconds for full searches (default: 1800).
- `--tlim_step`: CP-SAT limit in seconds per subgroup step when using the ladder policy (default: 60).
- `--hybrid_cap`: CP-SAT limit in seconds on the first hybrid maxodd attempt (default: 10).
- `--search_best`: Looks for the fastest policy of choosing the automorphism subgroup to generate a schedule.

## Paper results

`paper_results/BB_72_12_6/` contains the raw logical-error counts underlying the [[72,12,6]] figure, in JSON Lines. `localASC` is our depth-7 schedule, and `official` is the schedule of Bravyi et al. Runs at the same `p` are pooled when plotting. The code is in `codes/BB_Codes/BB_72_12_6/`.

## Citation

If you use this code or methodology in your research, please cite our work:

```bibtex
@misc{rocha2026localautomorphismawaresyndromecompilation,
      title={Local Automorphism-Aware Syndrome Compilation for General Quantum LDPC Codes}, 
      author={Eugenio Durazo Rocha and Olai Å. Mostad and Hsuan-Yin Lin and Eirik Rosnes},
      year={2026},
      eprint={2609.40319},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2609.40319}, 
}
```
