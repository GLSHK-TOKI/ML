# Template for Python3.11 Package

## Getting Started

#### 1. Edit `setup.py`

1. Replace `pvt` with your PVT name.
2. Replace `packagename` with the module name. It should be unique across your
    team.
3. Update the `description`.

#### 2. Edit `src/pvt_packagename` directory

1. Replace `pvt` with your PVT name.
2. Replace `packagename` with the module name. It should be unique across your
    team.

#### 3. Start development

1. Create a Virtual Environment
If your machine only has Python 3.11, you can create a virtual environment using: `python -m venv .venv`
To specify Python 3.11: `python3.11 -m venv .venv`
2. Activate the Virtual Environment: `source .venv/bin/activate`
3. Install the dependencies: `pip install`
4. `tests/example_test.py` contains an example test case. You should replace it with yours.

#### 4. Prepare to publish the module

1. Run the test cases (`python -m pytest`).
2. Update `setup.py` > `version`. Publishing the same version more than once
    would cause **unexpected failure** when the others install the module.
3. Update `README.md` (this file) to guide the others to use the module.
