#!/bin/bash

# Set the Python interpreter to use the virtual environment
export PYTHON_EXECUTABLE=/venv/bin/python

# Check if the first argument is 'runserver'
if [ "$1" = "runserver" ]; then
    # If it is, run the Django development server
    python manage.py runserver 0.0.0.0:8000
elif [ "$1" = "ipython" ]; then
    # If it's ipython, run ipython
    ipython
else
    # If not, execute the command as is
    exec "$@"
fi
