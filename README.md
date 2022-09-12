# shell-analysis-fenicsx

A Python module for shell analysis based on the Reissner-Mindlin plate theory and implemented with [FEniCSx](https://fenicsproject.org/), the next generation of the automated PDE solver FEniCS. We exploit an unconventional locking-free shell formulation presented by [Campello et al.](https://doi.org/10.1007/s00466-003-0458-8) with extension to use quadrilateral elements. This formulation has also been implemented with the legacy FEniCS by Jeremy Bleyer in his tutorial on [Linear Shell Model](https://comet-fenics.readthedocs.io/en/latest/demo/linear_shell/linear_shell.html), which has inspired the development of this module in many ways. The generic formulation and implementation make it convenient to be applied to different applications and also interchangeable with triangular and quadrilateral elements. Examples contain validation test cases from shell obstacle course and also more applied problems including structural analysis on different aircraft wing models.

# Get started:

In order to use this module, one should have the FEniCSx library installed first. There are several approaches to install FEniCSx based on the user's operation system and self preferences. For Windows users, you will need to use a Windows Subsystem for Linux (WSL) for the installation, and [Ubuntu](https://apps.microsoft.com/store/detail/ubuntu/9PDXGNCFSCZV?hl=en-us&gl=us) for Windows subsystem is preferred.

## Install FEniCSx
### Conda (MacOS, Ubuntu, Windows WSL)
Conda makes it easy for version control and intergration with other Python modules. You will need to install Conda first, and follow the [instructions](https://github.com/FEniCS/dolfinx#conda) to create an environment for FEniCSx.

### Ubuntu package for FEniCSx (Ubuntu, Windows WSL)
If you are using an Ubuntu system, you are able to install FEniCSx locally by the [Ubuntu PPA distribution](https://github.com/FEniCS/dolfinx#ubuntu-packages).


## Install the shell analysis module
Within the environment where you have FEniCSx installed, run:

```shell
git clone https://github.com/RuruX/shell_analysis_fenicsx.git
cd shell_analysis_fenicsx/
pip install -e .
```

## Test your installation

Now you can test your installation by running the examples under the `shell_analysis_fenicsx/examples/` directory. To run the tests, you should go into the directory, for example, `examples/scordelis_lo_roof/`, and run `scordelis_lo_roof.py`. If the installation was successful, you will be able to see the outputs of the simulation results, and you can also check the results through visualization by openning `solutions/u_mid.xdmf` in Paraview.
