# python311-gunicorn-flask-boilerplate

This repository serves as a boilerplate for a Model-View-Controller (MVC) design pattern application using `gunicorn`, `flask`, and some Flask-related libraries.

- [python311-gunicorn-flask-boilerplate](#python311-gunicorn-flask-boilerplate)
  - [Prerequisites](#prerequisites)
    - [Install Python 3.11](#install-python-311)
      - [Windows/ Linux](#windows-linux)
      - [MacOS](#macos)
  - [Setup](#setup)
    - [Set up the Virtual Environment](#set-up-the-virtual-environment)
      - [Create a Virtual Environment](#create-a-virtual-environment)
      - [Activate the Virtual Environment](#activate-the-virtual-environment)
    - [Install packages](#install-packages)
    - [Start the Server](#start-the-server)
    - [Run `pytest`](#run-pytest)
  - [Guidelines](#guidelines)
    - [Installing New Packages](#installing-new-packages)
    - [Security headers](#security-headers)
    - [Environment Variables](#environment-variables)
    - [MVC Model](#mvc-model)
    - [Test Cases](#test-cases)
    - [Database Connection](#database-connection)
    - [Flask Async](#flask-async)


## Prerequisites

### Install Python 3.11

#### Windows/ Linux
You can download Python 3.11 from the official Python website [here](https://www.python.org/downloads).

#### MacOS
Use the brew package manager to install Python 3.11:
```sh
brew install python@3.11
```

## Setup 

### Set up the Virtual Environment

#### Create a Virtual Environment
If your machine only has Python 3.11, you can create a virtual environment using:
```sh
python -m venv .venv
```

To specify Python 3.11:
```sh
python3.11 -m venv .venv
```

#### Activate the Virtual Environment
```sh
source .venv/bin/activate
```

### Install packages
```sh
pip install -r requirements.txt
```

### Start the Server
You may customise the configuration for your project during deployment as well. See Config gunicorn command arguments at the bottom.
```sh
gunicorn -t 120 -w 4 -b 0.0.0.0:5000 app.main:app --reload
gunicorn -c gunicorn_config.py app.main:app --reload
```

### Run `pytest`
```sh
python -m pytest 
```

## Guidelines

### Installing New Packages
Output installed packages and save as requirements.txt:
```sh
pip freeze > requirements.txt
```

### Security headers
Customize the configuration of Talisman security headers by referring to the official [Flask-Talisman documentation](https://github.com/GoogleCloudPlatform/flask-talisman).

### Environment Variables
Place environment variables and configurations in `config.py` and introduce them to Jenkins using either the `update-env` or `secret-update` pipelines.

### MVC Model
Follow the MVC design pattern for best practice. Store business logic in the `app/controllers` directory, routing-related code in the `app/routes` directory, and models in the `app/models` directory.

### Test Cases
Write test cases in the `tests/` directory and run them using:
```sh
python -m pytest
```

### Database Connection
To connect your Flask application to a database, install `sqlalchemy` and add the necessary models, controllers, and routes.

### Flask Async
To use asynchronous features in Flask, install `flask[async]` and modify the necessary code in the routes.

### Config gunicorn command arguments
For gunicorn to start the app using dedicted parameters
e.g. gunicorn -k uvicorn.workers.UvicornWorker app.main:app

Can use Jenkins pipeline(update-env) to update an env var 'GUNICORN_CMD_ARGS' to related values.
Sample value: -k uvicorn.workers.UvicornWorker

If prefer using config file, use `-c gunicorn_config.py` and specify the parameters in the config file.
ref: https://docs.gunicorn.org/en/stable/settings.html#config-file
