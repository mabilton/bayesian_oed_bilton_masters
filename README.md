# bayesian_oed_bilton_masters

## Description

This repository contains all of the code required to replicate my (Matt Bilton’s) Masters thesis results on Bayesian Optimal Experimental Design (OED). The code in this repositry depends on four Python packages which I wrote over the course of my Masters:
1. [`arraytainers`](https://github.com/MABilton/arraytainers), which are basically dictionaries or lists of Numpy arrays which act like arrays under some circumstances (e.g. when being added together, when being acted on by a Numpy function), but which like dictionaries/list under other circumstances (e.g. when accessing an element using a key).
2. [`approx_post`](https://github.com/MABilton/approx_post), which is used to create amortised variational inference posterior approximations.
3. [`oed_toolbox`](https://github.com/MABilton/oed_toolbox), which just provides some simple wrapper routines to compute and optimise some common OED criteria.
4. [`surrojax_gp`](https://github.com/MABilton/surrojax_gp), which is used to create simple Gaussian Process (GP) surogate models. 

## Directory Structure

There are five items of note within the main repository
1. Folders named `chapter_i`, each of which contains the code and figures corresponding to Chapter `i` of the thesis
2. The `fenics_models` folder, which contains all of the code which. Unlike the `chapter_i` folders, the code in this folder **must be run in the `dolfinx/lab` Docker container**; we’ll explain how to do this shortly.
3. `computation_helpers.py`, which containers helper functions for computations repeated across different notebooks (e.g. computing approximate posterior distributions through simple quadrature)
4. `plotting_helpers.py`, which contains helper functions to produce plots repeated across different notebooks (e.g. plotting multiple probability distributions against one another)
5. `requirements.txt`, which contains a list of the dependencies one must install to run the code in this repository. Importantly, these do not contain the dependencies required to run the code in the `fenics_models` folder: once again, the code in this folder must be run in the dolfinx Docker container.

Within each of the `chapter_i` folders, as well as within the `fenics_models` folder, one can find:
1. A series of Jupyter notebooks along with the the `json` data produced by those notebooks. Each corresponds to a particular piece of analysis performed within the chapter, and are numbered to indicated the order in which they were initially run. Although the notebooks have a definite ordering to them, they don’t need to be re-run in this same order since the outputs produced by each notebook (which may be required for other notebooks to run) have already been saved. For example, notebook `[3]` in the `chapter_5` folder requires access to the `nonlinear_beam_gp.json` file produced by notebook `[4]` in the `chapter_4` folder; this data, however, has already been saved within the `chapter_4` folder.
2. A `figs` folder, which contains all of the raw images produced by the Jupyter notebooks in that folder, along with figures created by `combining’ these raw images, and `hand-drawn’ images created using Inkscape. The figures in these `fig` folders are organised into further subcategory folders.

In addition to what was previously mentioned, the `fenics_models` folder also contains:
1. `fenics_helpers.py`, which defines a series of helper functions used by the notebooks in this folder, 
2.  A `data` folder, which contains all the data produced by the notebooks in this folder.

## Set-up

The set-up steps required to run the code in this repositry depend on whether you wish to run the Fenics code included in the `fenics_models` folder or not:

1. If you **don’t** want to run the Fenics code, then you **don't** need to worry about creating a Docker container.
2. If you **do** want to run the Fenics code, then you **will** need to install the appropriate Docker container.

Let's now go over these two situations in more detail.

### Everything but the Fenics Code

If you’re not interested in running the Fenics code, it’s sufficient just to install the requirements listed in the `requirements.txt` file; this can be done by navigating your terminal to the repository and executing:
    ```
    pip install -r requirements.txt
    ```
Importantly, since this repository uses the [Jax package](https://github.com/google/jax) (which you can read about [here](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)), you need to make sure your system is capable of installing Jax, which is described in detail [here](https://github.com/google/jax#installation). Put simply, running Jax basically requires you to use a Linux system. If you’re using a Windows-based system, we highly recommend you install Windows Subsystem for Linux (WSL), which will provide you with a light-weight version of Ubuntu within which you can run this repository’s code; instructions on how to install WSL can be found [here](https://docs.microsoft.com/en-us/windows/wsl/install).

### Everything, Including the Fenics Code

To run the Fenics code included in this repository, you’ll need to use the `dolfinx/lab` Docker image, which can be found [here](https://hub.docker.com/r/dolfinx/lab). Unfortunately, the Fenics project has a habit of regularly updating the dolfinx API (which you can read about [here](https://docs.fenicsproject.org/dolfinx/main/python/)) in a backwards incompatible manner. Consequently, we cannot guarantee that the code in the `fenics_models` folder will work with the current version of the `dolfinx/lab` image. To get around this, we’ve created our own Docker image which contains the appropriate version of the `dolfinx/lab` image to run the `fenics_models` code; this image can be found [here](https://hub.docker.com/r/mabilton/bayesian_oed_bilton_masters). This image also contains all of the dependencies listed in the `requirements.txt` file. Although the instructions to install this Docker image can be found in the `README` of the aforementioned Docker Hub repository, we’ll also give them here:
1. Install Docker; instructions on how to do this can be found [here](https://docs.docker.com/desktop/#download-and-install).
2. Clone this repository by running:
   ```
   git clone https://github.com/MABilton/bayesian_oed_bilton_masters
   ```
3. Navigate your terminal to inside of the pulled repository.
4. Run the command: 
   ```
   docker run --init -ti -p 8888:8888 -v "$(pwd)":/root/shared mabilton/bayesian_oed_bilton_masters:latest
   ```
   This will download the Docker image, create an accompanying container, and then launch that container. Note that Docker must be open for this command to work.
5. Open the Jupyter notebook link which appears in the terminal. If the first link doesn’t work, try using the second.
6. Run as many of the notebooks as you see fit – no other installations should be required.
7. Once you’re done, the container can simply be closed with `Ctrl + C` in your terminal.
8. Should you want to open the container again from where you left off, just run the command:
   ```
   docker start -i CONTAINER_ID
   ```
   where `CONTAINER_ID` is the ID of the Docker container created in Step 3. To find the ID of your container, execute in your terminal:
   ```
   docker ps -a
   ```
   The `CONTAINER_ID` will be shown in the first column next to the name of the image (i.e. `mabilton/bayesian_oed_bilton_masters:latest`)