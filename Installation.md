# Setup

Let us start by setting up your programming environment. Once you are finished, you should have everything installed you will need for this course.

Python 3 and virtual environments (conda) will be used for this class. Why use virtual environments? They are often used in large software projects, and for your project at the end of the semester, you will be expected to manage it using a virtual environment. Although not required, we highly recommend that you use a virtual environment for your homework assignments to get used to it before your project.

We *highly recommend* that you use Anaconda package manager (this is also what is explained below), as it makes installing packages much easier. We ran into issues when installing with package managers such as Pip and Homebrew, but you are still welcome to try those options at your own risk.

## Install Anaconda for Python3
Download [Anaconda](https://docs.anaconda.com/anaconda/user-guide/faq/) if you haven't already (the latest version of python3 is recommended)

## Setup without a Virtual Environment
Anaconda should already have all the packages you need by default except for openCV and dotenv. So once you have it installed, open an **Anaconda command prompt**, and run:
```
conda install -c conda-forge opencv
conda install -c conda-forge python-dotenv
```
If for whatever reason this does not work, you will have to use pip:
```
pip install opencv-python
pip install python-dotenv
```

If you want to use ipython, it should (probably) be updated with the same libraries you install with conda install. If any are missing, you will have to pip install them from within ipython. There are also some useful commands for ipython at the bottom of the Virtual Environment section that are worth checking out.

## Special Windows Instructions
If you are using a windows machine, opening an **Anaconda command prompt** is a bit less intuitive (especially if you don't want to set it as a path environment variable). If you have anaconda downloaded, you should be able to use Anaconda Prompt (it comes installed with anaconda). If you want to use the Anaconda shell from Visual Studio code, you need to enter the following command into the terminal:  
`c:\WINDOWS\System32\cmd.exe "/K" C:\Users\dylcm\Anaconda3\Scripts\activate.bat C:\Users\dylcm\Anaconda3`

Make sure that you replace the two `dylcm` bits above with whatever user is on your machine.

## Setup for a Python3.7 Virtual Environment

The rest of this tutorial will guide you through running a virtual environment. This will especially be necessary if you are running a different version of python than (the latest-ish) python3.

Open up an **Anaconda command prompt**, and run 
```
python --version
```

If you do not have the right version, you will have to manage separate Python versions. Anaconda does this through [environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Go to this link to troubleshoot if you have problems with your virtual environment.

1) Enter the commands below to create a new environment installed with the libraries you need for the homeworks (installs opencv4 or greater):
```
conda create --prefix ./virtual python=3.7 ipython matplotlib numpy Flask pandas scikit-learn
conda install -c conda-forge python-dotenv
conda install -c conda-forge opencv
```

Note: Make sure the name of the folder for your virtual environment is either "virtual" or "venv" since those filenames are included in the .gitignore . You probably don't want your repo to include your virtual environment folder (it is very large).

2) To activate this environment:
```
conda activate ./virtual
```

2.5) To deactivate this environment:
```
conda deactivate
```

3) To use ipython:
```
To enter: ipython
To exit: cntrl-d
```

3.5) If you want to use ipython in your virtual environment, you may need to pip install some libraries, probably opencv and dot-env.

4) Helpful ipython commands:
```
To install a package into ipython: pip install packagename
To see installed ipython packages: pip list
To use a conda/console command from ipython: ! conda list
```

5) Helpful conda commands:
```
To download a package w/conda: conda install packagename
To see installed conda packages: conda list
To see all of your conda environments: conda info --envs
```

6) Since you created the environment locally, you can just delete the `virtual` file to delete the environment.


## Troubleshooting & Checking your Python Version
If you think your installation has gone FUBAR, you can always uninstall it and start over. You can find instructions for this online ( [uninstalling anaconda](https://docs.anaconda.com/anaconda/install/uninstall/) ).

Open up an **Anaconda command prompt**, and try the following:  
`conda info`: Shows you the version of Python & Anaconda that you are running. Make sure that the you have Python 3.6 or higher.  
`python --version`: Shows you the version that is used whenever you run a .py file with `python <file.py>`. It should be Python 3.6 or higher, however...  
If the above command is showing a different version of python (thanks Apple), try:  
`conda run -n base python --version`. This should let you run python files from the version of Python you downloaded with Anaconda. If you are using a virtual environment, replace `base` with the name of that environment.