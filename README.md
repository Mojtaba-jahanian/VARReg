# VARReg: Variance-Based Adaptive Regularization

This repository contains the official implementation of the paper:

**"Adaptive Regularization Based on Neuron Activation Variance for Improved Generalization in Deep Neural Networks"**

## 📄 Overview

VARReg is a simple, biologically-inspired regularization technique that dynamically adjusts the L2 penalty of each neuron based on its activation variance. This allows highly active neurons to be less penalized (more flexible), while stable neurons are more strongly regularized (to prevent overfitting).

## 📁 Repository Structure

```
VARReg/
│
├── mlp_baseline/        # Shallow MLP experiments
├── mlp_deep/            # Deep MLP (3-layer) experiments
├── cnn_experiment/      # CNN-based experiments
├── utils/               # Regularization functions and dataset loaders
├── fig/                 # Figures and plots (to be added manually)
├── paper/               # PDF of the manuscript
│
├── demo.ipynb           # Jupyter demo notebook for MNIST + VARReg
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # This file
```

## 🧪 How to Run

Install required packages:

```bash
pip install -r requirements.txt
```

Run an experiment (example with CNN and adaptive regularization on CIFAR-10):

```bash
python cnn_experiment/train_cnn.py --dataset CIFAR10 --reg adaptive
```

Try baseline MLP:

```bash
python mlp_baseline/train_mlp.py --dataset MNIST --reg l2
```

## 📊 Results Summary

VARReg consistently improves or matches the performance of L1/L2 across MNIST, FashionMNIST, and CIFAR-10 in three architectures:

- Shallow MLP
- Deep MLP (512-256-128)
- CNN with 2 conv + 2 fc layers

See `paper/VARReg.pdf` and `fig/` for full results.

## 📜 License

This repository is licensed under the MIT License. See `LICENSE` for more information.

## ✉ Citation

If you use this work, please cite our paper:

```bibtex
@article{jahanian2025varreg,
  title={Adaptive Regularization Based on Neuron Activation Variance for Improved Generalization in Deep Neural Networks},
  author={Jahanian, Mojtaba and Karimi, Abbas and Osati Eraghi, Nafiseh and Zarafshan, Faraneh},
  journal={Journal of Machine Learning Research},
  year={2025}
}
```

---

For questions or collaborations, contact: mojtaba160672000@aut.ac.ir
