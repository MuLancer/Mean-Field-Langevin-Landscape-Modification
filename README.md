## Mean Field Langevin Dynamics: Noise Particle Gradient Descent with Landscape Modification

This project investigates particle-based optimization strategies for non-convex function minimization. We adapt the concept of Landscape Modification (LM) to Noisy Particle Gradient Descent (NPGD) and its variants, introducing a component current verage loss to dynamically update the threshold parameter \(c\) in LM. This mechanism leverages collective particle behavior to balance exploration and exploitation. We extend the framework to minimizing benchmark functions such as the Ackley function and conduct machine learning experiments on standard datasets MNIST and CIFAR-10 datasets, systematically comparing LM-enhanced particle based methods with standard SGD and Adagrad baselines with independent runs. Our results demonstrate that while LM offers limited advantage on simple tasks like MNIST, it provides tangible convergence benefits on more complex landscapes such as CIFAR-10 when paired with adaptive optimizers.

### Dependencies
- Python 3.10+
- `pandas`
- `numpy`
- `torch`
- `torchvision`
- `matplotlib`

Install via pip:
```bash
pip install pandas numpy torch torchvision matplotlib
```

### Directory Structure
```bash
Code/
├── CIFAR10.ipynb           # CIFAR-10 experiment
├── MNIST.ipynb             # MNIST experiment
├── MFL_experiments.ipynb   # Mean-field Langevin experiments
├── toy_experiments.ipynb   # Toy optimization problems
├── Data/                   # Dataset storage folder
```

### Landscape Modification Demo 
Run the following notebook to see a demonstration of landscape modification on optimization problems with multiple particles (m=3):
```bash
Code/LM_demo.ipynb
```


### Experiments
- Mean-Field Langevin Experiments
Replicates results from Ichizat (2022) on noise particle gradient descent and distribution matching. Includes LM-enhanced version for comparison.
```bash
Code/MFL_experiments.ipynb
```

- Toy Function Minimization
Simple experiments to illustrate function optimization using particle dynamics and landscape modification with visualizations.
```bash
Code/toy_experiments.ipynb
```

- Machine Learning experiments:
Experiments on CIFAR-10 and MNIST datasets.
```bash
Code/CIFAR10.ipynb
Code/MNIST.ipynb
```

