# Operational Space Control Barrier Functions

Code for *"Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions"* -- Daniel Morton and Marco Pavone

## Why should you use this?

- Do you have a robot arm?
- Do you want to keep it from crashing into things or generally behaving poorly?

If so, you should probably use this.

To be more specific, if you...
- Are controlling the robot based on operational-space (end-effector) motions
- Are concerned about the safety of your controller, which doesn't necessarily have safety guarantees (as in **teleoperation** or **learned policies**)
- Want to keep the robot performing "as best as possible" even when close to an unsafe situation

Then, OSCBF should be helpful. 

Even if you don't fit this criteria entirely, this will be helpful
- "But, I'm controlling joint-space motions -- does this still apply?"
  - The collision avoidance and set-invariance CBF constraints will still be useful!
- "But, I already have a well-tuned operational space controller! I don't want to replace that"
  - You can use OSCBF as a safety filter on top of this nominal controller!

For more details and videos, check out the [project webpage](https://stanfordasl.github.io/oscbf/) as well as the [CBFpy documentation](https://danielpmorton.github.io/cbfpy/).


## Installation

A virtual environment is optional, but highly recommended. For `pyenv` installation instructions, see [here](https://danielpmorton.github.io/cbfpy/pyenv).

```
git clone https://github.com/stanfordasl/oscbf
cd oscbf
pip install -e .
```

Note: This code will work with most versions of `jax`, but there seems to have been a CPU slowdown introduced in version `0.4.32`. To avoid this, I use version `0.4.30`, which is the version indicated in this repo's `pyproject.toml` as well. However, feel free to use any version you like.

## Examples

Check out the `examples/` folder for interactive demos in Pybullet

## Documentation

See the CBFpy documentation, available at [this link](https://danielpmorton.github.io/cbfpy)


## Citation
```
@article{morton2025oscbf,
        author = {Morton, Daniel and Pavone, Marco},
        title = {Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions},
        year = {2025},
        note = {Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025},
      }
```

