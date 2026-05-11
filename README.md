# Lookahead Branching for Neural Network Verification

This repository contains the implementation and benchmarks accompanying
*Lookahead Branching for Neural Network Verification*, to appear at 
IJCAI 2026. Lookahead branching is a general branching heuristic for 
branch-and-bound-based neural network verifiers that improves branching decisions by
simulating candidate splits to a specified depth and aggregating scores across the resulting
branches. This repository contains the source code and benchmarks for both
Marabou and α-β-CROWN.

If you use lookahead branching in your work, please use the following bibtex:

```bibtex
@inproceedings{davis2026lookahead,
  title     = {Lookahead Branching for Neural Network Verification},
  author    = {Davis, Liam and Zhou, Duo and Zhang, Huan and Katz, Guy and Barrett, Clark and Wu, Haoze},
  booktitle = {Proceedings of the 35th International Joint Conference on
               Artificial Intelligence (IJCAI)},
  year      = {2026}
}
```

The Marabou and α,β-CROWN forks bundled here are pinned to older upstream versions of each solver. 
In particular, α,β-CROWN has seen significant upstream changes, and we are currently building 
Marabou 3.0 from scratch. The bundled fork should be treated
as the snapshot evaluated in the paper rather than a maintained library. Once
new lookahead-branching implementations land on top of the current
generation of either solver, this repository will be updated to reflect
those artifacts.

## Instructions

### Repository layout

```
.
├── marabou/                Modified Marabou source tree
├── alpha-beta-crown/       Modified α,β-CROWN source tree
└── benchmarks/
    ├── nap/                235 NAP instances (.ipq)
    ├── nap.list            One instance per line
    ├── nn4sys/             4 ONNX models + 30 VNN-LIB properties
    ├── nn4sys.list         120 (model, property) pairs per line
    ├── mnist20x20/         500 MNIST 20×20 instances (.ipq)
    └── mnist20x20.list     One instance per line
```

### Building Marabou

Requires CMake ≥ 3.16, a C++17 compiler, and Boost / OpenBLAS / pybind11
(downloaded automatically by the build). Optional: Gurobi.

```bash
cd marabou
mkdir build && cd build
cmake ..
cmake --build . -j8
```

The binary is written to `marabou/build/Marabou`.

### Running Marabou experiments

Lookahead branching is enabled with `--lookahead-branching`. The two branching
heuristics evaluated in the paper are `babsr` and `pseudo-impact`.

#### NN4Sys (ONNX + VNN-LIB)

```bash
while read onnx vnnlib; do
    ./marabou/build/Marabou --verbosity=1 --lookahead-branching --branch babsr \
        "benchmarks/$onnx" "benchmarks/$vnnlib"
done < benchmarks/nn4sys.list
```

#### NAP (input queries)

```bash
while read ipq; do
    ./marabou/build/Marabou --verbosity=1 --lookahead-branching --branch babsr \
        --input-query "benchmarks/$ipq"
done < benchmarks/nap.list
```

#### MNIST 20×20 (input queries, native LP)

```bash
while read ipq; do
    ./marabou/build/Marabou --verbosity=1 --lookahead-branching --branch babsr \
        --lp-solver native --input-query "benchmarks/$ipq"
done < benchmarks/mnist20x20.list
```

To run the baseline (no lookahead) on any benchmark, drop the
`--lookahead-branching` flag.

### Configuring lookahead

Three constants in `marabou/src/configuration/GlobalConfiguration.cpp` control
the heuristic and are documented inline:

| Constant                    | Default | Role                                                                      |
|-----------------------------|---------|---------------------------------------------------------------------------|
| `NUM_LOOKAHEAD_BRANCHES`    | 20      | Upper bound on the preselect pool scored by the cheap polarity heuristic. |
| `NUM_LOOKAHEAD_CANDIDATES`  | 10      | Of that pool, how many are evaluated with the full lookahead procedure.   |
| `LOOKAHEAD_MAX_STACK_DEPTH` | 5       | Lookahead is only invoked when the SMT stack depth is below this value.   |

The recursion depth used *inside* `branchWithLookahead` (i.e. how many trial
splits deep we go when estimating phase fixes) is exposed as the runtime option
`MAX_LOOKAHEAD_DEPTH` (default 2) in `marabou/src/configuration/Options.cpp`.

### Setting up α,β-CROWN

```bash
cd alpha-beta-crown
conda env create -f complete_verifier/environment.yaml
conda activate alpha-beta-crown
```

The lookahead branching heuristic is implemented in
`alpha-beta-crown/complete_verifier/heuristics/lookahead.py` and dispatched
from `alpha-beta-crown/complete_verifier/heuristics/branching_heuristics.py`.

### Running α,β-CROWN experiments

The configs used in the paper live under
`alpha-beta-crown/complete_verifier/exp_configs/`. Selected configs:

| Benchmark            | Config                                                |
|----------------------|-------------------------------------------------------|
| CIFAR CNN-A (adv)    | `exp_configs/beta_crown/cifar_cnn_a_adv.yaml`         |
| CIFAR CNN-A (mix)    | `exp_configs/beta_crown/cifar_cnn_a_mix.yaml`         |
| CIFAR CNN-B (adv)    | `exp_configs/beta_crown/cifar_cnn_b_adv.yaml`         |
| MNIST CNN-A (adv)    | `exp_configs/beta_crown/mnist_cnn_a_adv.yaml`         |
| VNN-COMP'21 ResNet   | `exp_configs/vnncomp21/cifar10-resnet.yaml`           |
| VNN-COMP'24 CIFAR100 | `exp_configs/vnncomp24/cifar100.yaml`                 |
| VNN-COMP'24 TinyImg  | `exp_configs/vnncomp24/tinyimagenet.yaml`             |

Run a config with:

```bash
cd alpha-beta-crown/complete_verifier
python abcrown.py --config exp_configs/beta_crown/cifar_cnn_a_adv.yaml
```

The configs reference models and datasets via paths relative to
`complete_verifier/`; VNN-COMP benchmarks are downloaded by the upstream
α,β-CROWN tooling and are not bundled here.

### Reproducing the paper's tables

The depth, candidate-pool, and phase-fixing ablations correspond to the three
`GlobalConfiguration` constants above and the runtime `MAX_LOOKAHEAD_DEPTH`
option. Each ablation cell was produced by rebuilding Marabou with the
relevant value changed and running all three benchmark suites with both
`babsr` and `pseudo-impact`.
