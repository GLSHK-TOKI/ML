# https://docs.gunicorn.org/en/latest/configure.html
# Set the timeout value
timeout = 120

# Set the number of worker processes
workers = 4

# Bind to the specified address and port
bind = '0.0.0.0:5000'
