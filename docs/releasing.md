# Publishing and deployment

The Python package version starts at **0.1.0**. Its release tag is `v0.1.0` in
`rjn32s/gemmadecision`. The model remains the separate, pinned **v0.4.0** release
in `rajan2k/GemmaDecision-270M`. A package release does not retrain or republish
the model, and the wheel contains no weights or Hugging Face credential.

## CI

`.github/workflows/ci.yml` tests Python 3.11 and 3.12 on Linux. An isolated
`--no-deps` install with only Pydantic and HTTPX checks that imports stay lazy.
The normal package includes local inference, serving, and PydanticAI by default.
CI then installs CPU-only Torch before installing the complete package and test
dependencies, avoiding CUDA downloads. It runs mocked server/client and
PydanticAI tests plus small synthetic Torch tensor checks, then builds and
validates both distributions. Tests run with Hugging Face offline mode and do
not load model weights. Optional vLLM tests are skipped when unavailable.
CI is not a GPU speed benchmark.

## One-time PyPI account setup

Sign in as the intended PyPI owner, `rjn32s`, and use
[PyPI publishing settings](https://pypi.org/manage/account/publishing/) to add a
pending GitHub trusted publisher with these exact values:

| Field | Value |
|---|---|
| PyPI project | `gemmadecision` |
| GitHub owner | `rjn32s` |
| GitHub repository | `gemmadecision` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

Create the `pypi` environment in that repository's GitHub Settings →
Environments. The workflow scopes OIDC permission to its publishing job and
does not require a long-lived PyPI token. Configuring a pending publisher does
not reserve the name; the project is created by the first successful upload.
These are the [official pending-publisher steps](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [publishing workflow](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

For an existing PyPI project, add the same publisher from that project's
Publishing settings instead. If a repository/environment configuration or
PyPI login is still required, the package is built and release-ready but is
**not yet published**. A successful wheel build is not evidence of a PyPI upload.

## Release a version

1. Verify the intended GitHub identity is `rjn32s`, update the package version,
   and ensure the complete CI and relevant real-model compatibility checks pass.
2. Commit and push the release source, then tag that commit `v0.1.0` (or the
   matching next version). Never move an existing release tag.
3. Publish the GitHub release for that tag. Alternatively run **Publish to
   PyPI** manually and supply the existing tag. Do one or the other for a given
   version, since PyPI will reject a duplicate distribution upload.
4. The workflow checks that the tag matches `pyproject.toml`, runs tests,
   builds the distributions, validates metadata, and uploads via OIDC with
   package attestations. Its publishing job executes no repository code.
5. Verify the published PyPI metadata and perform a fresh client install of the
   exact version before announcing availability. Preserve measured performance
   evidence separately from marketing claims.

## CPU container

Build the image from the package source directory:

```sh
docker build -t gemmadecision:0.1.0 .
```

Set `GEMMADECISION_API_KEY` in your shell or secret manager, then run:

```sh
docker run --rm \
  -p 127.0.0.1:8700:8700 \
  -e GEMMADECISION_API_KEY \
  -v gemmadecision-models:/models \
  gemmadecision:0.1.0
```

The container runs as UID 10001. Its first startup downloads the pinned public
model into the writable `/models` volume, then verifies the inference files.
It needs no HF token for this model. `/ready` becomes healthy after loading;
the image health check allows time for the initial download. Requests use the
configured API key. The Rust HTTP runtime is Granian; transformer math runs in
the selected tensor backend. This image uses **CPU-only** PyTorch and does not
claim GPU throughput. Use the documented vLLM path or a CUDA runtime for GPU
serving, rather than adding `--gpus` to this CPU image.
