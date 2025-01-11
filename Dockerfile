FROM python:3.13.1-bookworm AS assisted-rag-builder

# Create a virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

# Activate virtual environment
RUN . /venv/bin/activate

# Create the /app/docker directory
RUN mkdir -p /app/docker

# COPY command uses the directory containing the Dockerfile as the build context
# The path is relative to the build context (Dockerfile location)
COPY requirements.txt /app/requirements.txt

# Install Dependencies using the virtual environment's pip
RUN /venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# # # Node.js and NPM Stage # # #
FROM node:20-bookworm-slim AS node-builder

# Install Node.js and npm in a specific directory
WORKDIR /node

# Copy package.json and package-lock.json
COPY package.json package-lock.json* ./

RUN npm install -g npm@latest

# # # Runtime Stage -- Copy over dependencies # # #
FROM python:3.13.1-bookworm AS assisted-rag

# Copy the python virtual environment from the builder stage
COPY --from=assisted-rag-builder /venv /venv

# Copy Node.js and npm from the node-builder stage
COPY --from=node-builder /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node-builder /usr/local/bin/node /usr/local/bin/
COPY --from=node-builder /usr/local/bin/npm /usr/local/bin/
COPY --from=node-builder /node /node

# Set the PATH environment variable to use the virtual environment
ENV PATH="/venv/bin:/node/node_modules/.bin:$PATH"

# Set PYTHONPATH to include the application directory
ENV PYTHONPATH="/app:$PYTHONPATH"

# Set the working directory
WORKDIR /app

# Activate virtual environment
RUN . /venv/bin/activate

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy the rest of the application code into the container
COPY . /app/

ENTRYPOINT [ "/entrypoint.sh" ]
