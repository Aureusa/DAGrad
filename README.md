# 🚀 DAGrad

**Modular DAG-based workflows with end-to-end autodiff and gradient optimization.**


---

## ✨ What is DAGrad?

DAGrad is a lightweight framework for building differentiable computational workflows from reusable blocks.
It combines:

- 🧱 **Composable `Block` units** for modular logic
- 🕸️ **Workflow execution** in sequential or graph (DAG) mode
- 🔁 **End-to-end autodiff** via PyTorch tensors
- 📉 **Gradient-based optimization** with standard Torch optimizers

The goal is to make it easy to prototype and train structured, interpretable computational pipelines while keeping the code simple and explicit.

---

## 🎯 Framework Goals

DAGrad aims to be:

- **Modular**: Break models into clear, reusable components
- **Explicit**: Define data flow with named nodes/ports in graph mode
- **Differentiable**: Keep everything compatible with PyTorch autograd
- **Practical**: Train workflows directly with your favorite optimizers and loss functions

---

## ✅ Current Capabilities

- Define custom blocks by subclassing `Block`
- Register trainable parameters with `add_param(...)`
- Compose blocks into a `Workflow`
- Run workflows in:
  - **Sequential mode** (insertion order)
  - **Graph mode** (explicit `connect_input` / `connect` wiring)
- Expose multiple named outputs using `set_outputs(...)`
- Move full workflows to CPU/GPU via `.to(device)`

---

## 📦 Installation

### Editable install (recommended for development)

```bash
pip install -e .
```

### With development extras (Jupyter)

```bash
pip install -e ".[dev]"
```

### If using a conda env (example: `degrad`)

```bash
conda activate degrad
pip install -e .
```

---

## 🧠 Quick Start

```python
from dagrad.engine.block import Block
from dagrad.engine.workflow import Workflow

class LinearLike(Block):
	def __init__(self):
		super().__init__()
		self.add_param(1.0, symbol="w", trainable=True)
		self.add_param(0.0, symbol="b", trainable=True)

	def execute(self, x):
		return self.w * x + self.b

class MyWorkflow(Workflow):
	def __init__(self):
		super().__init__()
		self.add_block(LinearLike())

workflow = MyWorkflow()
```

Train with standard PyTorch:

```python
import torch

x = torch.linspace(-1, 1, steps=64)
y_gt = 2.0 * x + 0.5

opt = torch.optim.Adam(workflow.parameters(), lr=1e-2)
loss_fn = torch.nn.L1Loss()

for _ in range(200):
	y_pred = workflow.run(x)
	loss = loss_fn(y_pred, y_gt)
	opt.zero_grad()
	loss.backward()
	opt.step()
```

---

## 🕸️ Graph Mode at a Glance

Graph mode lets you wire blocks with named ports and retrieve structured outputs.

Core APIs:

- `add_block(block, key="node_name")`
- `connect_input("input_name", "node", dst_input="arg")`
- `connect("src", "dst", src_output="out", dst_input="arg")`
- `set_outputs({"name": ("node", "port")})`

This is useful when workflows are **non-linear**, **multi-branch**, or need **intermediate outputs** for monitoring/debugging.

---

## 📓 Example Notebook

See the showcase notebook for both sequential and graph workflows:

- `examples/showcase_simple_workflow.ipynb`

---

## 📄 License

This project is licensed under the **MIT License** ✅


See [LICENSE](LICENSE) for the full text.

