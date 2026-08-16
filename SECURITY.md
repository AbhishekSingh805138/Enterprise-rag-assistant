# Security

## Reporting

Report suspected vulnerabilities privately to the maintainers rather than
via a public issue.

## Dependency scanning

`pip-audit` runs on every push and pull request and **fails the build** on
any known vulnerability that is not listed below.

## Accepted vulnerabilities

Each entry is a decision with an owner and an exit condition, not an
oversight. Re-review at every dependency bump and whenever a fix version
appears. The corresponding `--ignore-vuln` flags live in
`.github/workflows/ci.yml`; the two lists must be kept in step.

### PYSEC-2026-311 — `chromadb` 1.x, pre-authentication code injection

| | |
|---|---|
| **Severity** | Critical |
| **Fix available** | None for any 1.x release |
| **Exposure** | Requires network reach to the Chroma server port |
| **Status** | Mitigated by network isolation |

The Chroma service publishes **no host port**. Only `api` and `worker`
reach it, over the internal compose network as `chroma:8000`.

> Publishing that port re-opens a pre-authentication remote code execution
> path. Do not add a `ports:` mapping to the `chroma` service, and do not
> place it on a shared network with untrusted workloads.

Downgrading is not an option: a 0.6.x server is incompatible with the 1.x
client (it fails with an opaque `KeyError('_type')`). Remove this exception
when ChromaDB ships a patched release.

### PYSEC-2026-3046 / PYSEC-2026-3047 — `ragas`, arbitrary file read

| | |
|---|---|
| **Severity** | High |
| **Fix available** | 3047 in `0.3.0rc1` (prerelease); 3046 unfixed |
| **Exposure** | Multi-modal / image prompt handling only |
| **Status** | Not reachable in this project |

Both flaws are in image-and-text prompt handling. Evaluation here is
text-only, and `ragas` is an **offline evaluation dependency** — it is not
imported by the API or the ingestion worker, so it is absent from the
serving path entirely.

Remove this exception when `ragas` 0.3.x is stable and the LangChain import
breakage noted in `requirements.txt` is resolved.

### PYSEC-2026-2447 — `diskcache`, unsafe deserialization

| | |
|---|---|
| **Severity** | Medium |
| **Fix available** | None |
| **Exposure** | Requires write access to the cache directory |
| **Status** | Accepted |

Exploitation requires an attacker to already have filesystem write access
inside the container, at which point they have better options. `diskcache`
arrives transitively via `ragas` and is not used in the serving path.

Remove this exception when `diskcache` offers a safe serializer by default,
or when `ragas` stops depending on it.

## Deployment hardening

Settings the production compose file relies on:

- **No infrastructure ports on the network.** `chroma` and `kafka` publish
  nothing; `minio` binds to `127.0.0.1` only, reachable via an SSH tunnel.
  Only `api` and `ui` are exposed.
- **Authentication on.** `AUTH_ENABLED=true` with keys supplied through
  `API_KEYS` (or `API_KEYS_FILE`).
- **Deep health is authenticated.** `GET /health` stays public for load
  balancer probes; `?deep=true` reports internal topology and requires a
  key.
- **Secrets from files.** Any secret can be supplied as `<NAME>_FILE`
  pointing at a mounted path — the Docker and Kubernetes secret
  convention — which keeps credentials out of the process environment where
  `docker inspect` and crash dumps can reach them.
- **Debug off.** `DEBUG_MODE=false` so internal exception text is not
  returned to clients.

## Known gaps

Tracked, not yet closed:

- Uploaded documents are parsed without malware scanning or sandboxing.
  PDF and DOCX parsers are a recognised exploit surface.
- Document text is embedded without PII screening; vectors may contain PII
  even though the answer-time filter redacts it from responses.
- Retrieval has no per-user access control — any authenticated caller can
  query any department. Do not enable multi-department access until this
  lands.
