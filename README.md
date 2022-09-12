# shell-analysis-fenics

A Python module for general shell analysis based on the Reissner-Mindlin plate theory, which heavily utilizes FEniCSX, the next generation of the automated PDE solver FEniCS. The generic formulation and implementation make it convenient to be applied to different applications and also interchangeable with triangular and quadrilateral elements. A collection of classic example problems has been implemented to verify the accuracy of the results, and other applications like structural analysis on an aircraft wing have also been tested.


# Get started:

In order to use this module, one should have the [DOLFINx](https://github.com/FEniCS/dolfinx) library installed first. There are several approaches to install DOLFINx based on the user's operation system and self preferences.

## Install shell module with local DOLFINx library
If you are using a Linux system, you are able to install DOLFINx locally by either [building from source](https://github.com/FEniCS/dolfinx#from-source) or installing its [PPA distribution](https://launchpad.net/~fenics-packages/+archive/ubuntu/fenics), which is less frequently updated.

## Install shell module with Docker container

- Open a new terminal window in the directory where you would like to store the shell module library. Then from the [`docker`](https://github.com/RuruX/shell_analysis_fenics/tree/docker) branch, download the code as a zip and extract it to be a directory. Here you have the docker-friendly version of the shell module.
- Install Docker by downloading the [Docker desktop](https://hub.docker.com/editions/community/docker-ce-desktop-mac) app on your computer.
- Now you should have the `docker` commands available on your computer. Here we are going to activate the Docker container for DOLFINx. 
    In the directory `shell_analysis_fenics-docker`, run:
    
    ```bash
    docker run -ti -v $(pwd):/home/shared -w /home/shared/ --rm dolfinx/dolfinx
    ```
    If it is your first time using it, it will take some time to pull the Docker image of DOLFINx remotely. After it's done, you would be in the directory called `/home/shared/`, which actually contains all of the files under the original `/shell_analysis_fenics-docker` directory. From there, you should have DOLFINx installed successfully.
- The final step to complete the installation is to install the Python module of `shell_analysis_fenics-docker`, by running:

    ```bash
    pip3 install .
    ``` 
    from the `/home/shared/` directory.

## Test your installation

Now you can test your installation by running the examples under the `shell_analysis_fenics-docker/examples/` directory. To run the tests, you should go into the directory, for example, `examples/CRM/`, and run `CRM_SI.py`. If the installation was successful, you will be able to see the outputs of the simulation results, and you can also check the results through visualization by openning `solutions/u_mid.xdmf` in Paraview.
