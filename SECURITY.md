# Security policy

## Supported versions

PocketWorld is currently in the `0.1.x` research-prototype series. Security fixes are applied to the latest commit on `main`.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting flow under the repository's **Security** tab. Do not open a public issue for vulnerabilities involving dependency compromise, unsafe model files, browser execution, or arbitrary file access.

Include the affected commit, reproduction steps, impact, and any suggested mitigation. You should receive an acknowledgement within seven days. A fix and disclosure timeline will be coordinated through the private advisory.

## Model and checkpoint safety

PyTorch checkpoints are loaded with `torch.load` and must be treated as executable, trusted artifacts. Only load checkpoints published by this repository's official GitHub Releases or produced locally from reviewed code. ONNX files should likewise be verified against the release checksum before deployment.
