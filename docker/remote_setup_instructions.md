# Docker Compose Setup for Local and Remote OpenPI Server

This setup supports both of these cases with a single `docker-compose.yaml`:

1. Run `openpi_server`, `eval`, and `model` on the same machine.
2. Run `openpi_server` on a remote machine, while `eval` and `model` run locally.

## How it works

The `openpi_server` service uses `network_mode: host`, so it is reachable on the Docker host's network rather than through Docker Compose service DNS.

The `model` service therefore connects to:

- `host.docker.internal` by default, which reaches the local Docker host
- `OPENPI_HOST_IP` when you want to use a remote OpenPI server

The relevant part of the configuration is:

```yaml
model:
  extra_hosts:
    - "host.docker.internal:host-gateway"
  command: --ros-args -p policy:=aic_example_policies.ros.RunOpenPIBase_latest -p openpi_host:=${OPENPI_HOST_IP:-host.docker.internal}
```

## Scenario 1: All three services on the same machine

Start everything together:

```bash
docker compose up
```

In this mode:

- `openpi_server` runs on the host network
- `model` connects to `host.docker.internal`
- `eval` and `model` still communicate with each other over the Compose network

## Scenario 2: Remote `openpi_server`, local `eval` and `model`

On the remote machine, start only `openpi_server`:

```bash
docker compose up openpi_server
```

On the local machine, start `eval` and `model` and point `model` at the remote machine:

```bash
OPENPI_HOST_IP=YOUR_REMOTE_IP_ADDRESS docker compose up eval model
```

Replace `YOUR_REMOTE_IP_ADDRESS` with the reachable IP address or hostname of the remote machine running `openpi_server`.

## Scenario 3: Only run `openpi_server`

If you only want the OpenPI server on a machine:

```bash
docker compose up openpi_server
```

## Notes

- The Compose network is not marked `internal`, so `model` can reach a remote OpenPI host when needed.
- `host.docker.internal` is explicitly mapped for Linux using Docker's `host-gateway` support.
- If the remote setup still cannot connect, make sure the OpenPI server is listening on the expected port and that firewalls allow inbound traffic.
