# Operational Space Control Barrier Functions

Code for *"Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions"* -- Daniel Morton and Marco Pavone

## Installation

A virtual environment is optional, but highly recommended. For `pyenv` installation instructions, see [here](https://danielpmorton.github.io/cbfpy/pyenv).

```
git clone https://github.com/stanfordasl/oscbf
cd oscbf
pip install -e .
```

Note: This code will work with most versions of `jax`, but there seems to have been a CPU slowdown introduced in version `0.4.32`. To avoid this, I use version `0.4.30`, which is the version indicated in this repo's `pyproject.toml` as well. However, feel free to use any version you like.


## Documentation

See the CBFpy documentation, available at [this link](https://danielpmorton.github.io/cbfpy)

## Examples

Check out the `examples/` folder for interactive demos in Pybullet

## Citation
```
@article{morton2025oscbf,
        author = {Morton, Daniel and Pavone, Marco},
        title = {Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions},
        year = {2025},
        note = {Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025},
      }
```

