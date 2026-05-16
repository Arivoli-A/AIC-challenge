# Docker Compose Setup for Local and Remote OpenPI Server

This document outlines how to use the `docker-compose.yaml` file to run your services in various configurations, supporting both local and remote deployment of the `openpi_server`.

## Configuration Overview

The `docker-compose.yaml` is configured to allow the `model` service to connect to the `openpi_server` either locally (by default) or remotely (via an environment variable).

The relevant part of the `model` service configuration is:
```yaml
  model:
    # ...other configurations...
    command: --ros-args -p policy:=aic_example_policies.ros.RunOpenPIBase_latest -p openpi_host:=${OPENPI_HOST_IP:-openpi_server}
    # ...other configurations...
```
-   If the `OPENPI_HOST_IP` environment variable is set, its value will be used as the `openpi_host`.
-   If `OPENPI_HOST_IP` is not set, `openpi_server` will be used as the hostname, which will resolve to the `openpi_server` service defined in the same `docker-compose.yaml` file if it's running locally.

## Usage Scenarios

Here are the different ways you can use the `docker-compose.yaml` file:

### 1. Run all three containers locally (openpi_server, eval, model)

This is the default and most straightforward way to run your entire setup on a single machine. All three services (`openpi_server`, `eval`, and `model`) will be started, and the `model` service will automatically connect to the `openpi_server` service via Docker's internal networking.

**Command:**
```bash
docker compose up
```
(Alternatively, you can explicitly list all services: `docker-compose up openpi_server eval model`)

### 2. Run `openpi_server` remotely and `eval` and `model` locally

This scenario is useful when you want to offload the `openpi_server` (e.g., to a powerful machine with a TPU) while running the `eval` and `model` services on your local machine.

**Steps:**

#### On your remote machine:
Start only the `openpi_server` service.

**Command:**
```bash
docker compose up openpi_server
```

#### On your local machine:
Start the `eval` and `model` services. You need to tell the `model` service the IP address or hostname of your remote `openpi_server` by setting the `OPENPI_HOST_IP` environment variable.

**Command:**
```bash
OPENPI_HOST_IP=YOUR_REMOTE_IP_ADDRESS docker compose up eval model
```
**Important:** Replace `YOUR_REMOTE_IP_ADDRESS` with the actual IP address or hostname of your remote machine where the `openpi_server` is running.

### 3. Run only `openpi_server` on a remote machine

If your goal is solely to run the `openpi_server` on a remote machine without involving the `eval` and `model` services locally, follow these steps.

**Steps:**

#### On your remote machine:
Start only the `openpi_server` service.

**Command:**
```bash
docker-compose up openpi_server
```

#### On your local machine:
No action is required if you only intend to run the `openpi_server` remotely.
