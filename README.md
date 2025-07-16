# Template to setup a new project developing packages

docker run -it --volume .\src\:/app/src --volume .\outer_assets\:/app/outer_assets --net=host --gpus=all -e DISPLAY=192.168.233.1:0.0 computervisionbasicfunctions:latest /bin/bash
cd /app
source /app/.venv/bin/activate

## Open PLC
ENCODER_VALUE_MAX = 65535
ENCODER_VALUE = 21910 --> 5cm
ENCODER_VALUE = 43820 --> 10cm
ENCODER_VALUE = 87640 --> 20cm

## Python env

```ps
# Windows
# -------

cd <path_to_project_folder>
py -m venv .venv
.venv\Scripts\activate.ps1

# why command `py` and not `python`: https://docs.python.org/3/using/windows.html#getting-started

# virtual environments with standard library: https://docs.python.org/3.12/library/venv.html
```


## project layout example

- https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#src-layout
- https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/#src-layout-vs-flat-layout

```ps
project_root_directory
├── pyproject.toml      # AND/OR setup.cfg, setup.py
├── ...
└── src/
    └── mypkg/
        ├── __init__.py
        ├── ...
        ├── module.py
        ├── subpkg1/
        │   ├── __init__.py
        │   ├── ...
        │   └── module1.py
        └── subpkg2/
            ├── __init__.py
            ├── ...
            └── module2.py
```


## pyproject.toml file

- https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- https://packaging.python.org/en/latest/specifications/pyproject-toml/
- https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html


## installation of local projects as packages

- https://pip.pypa.io/en/stable/topics/local-project-installs/#local-project-installs

- regular install (pyproject.toml - based) as package 

    ```ps
    rm -r build
    pip install . 

    # creates build folder, then installs to .venv\site-packages (inspect __path__ of package for info)
    # clearing build folder before, preserves of hard do unterstand bugs!
    # I.e. after renaming of modules and rebuild, the build folder contains the old AND the new module
    ```

- editable install (pyproject.toml - based) as package

    ```ps
    pip install -e .  

    # editable install without build step
    ```

- How to inspect what is installed?
    - check `.venv\lib\site-packages\<package-name>`
    - run `pip list` or `pip freeze` to see installed packages and if they are editable


## FAQ

1. What if environment is crazy?  
    - delete ./.venv and setup new venv
    - install dependencies new in venv

2. how to package something else than .py-files?
    - https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html
    - https://setuptools.pypa.io/en/latest/userguide/datafiles.html
    - for example:
        - `./src/inner_assets` & `./outer_assets` are defined in `MANIFEST.in`
        - but only `./src/inner_assets` will be packaged, cause `./outer_assets` is outside of the package scope `src`

