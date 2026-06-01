# Protected `main` branch settings

Configure a GitHub branch protection rule or repository ruleset for `main`. The release workflows rely on the tagged commit having passed the same reusable CI workflow, while branch protection keeps unreviewed commits from entering the release branch.

## Pull request and push policy

- Require a pull request before merging into `main`.
- Require at least **one approving review**. Dismiss stale approvals when new commits are pushed.
- Restrict updates to `main` so contributors cannot push directly. Apply the restriction to administrators and automation unless a deliberately scoped release or maintenance actor needs an exception.
- Block force pushes and branch deletion.
- Require branches to be up to date before merging so the required checks evaluate the exact merge candidate.

## Required CI status checks

Require every blocking job exposed by **Python CI**:

- `test (3.10)`
- `test (3.11)`
- `test (3.12)`
- `lint`
- `type-check`
- `package-build`
- `container-smoke`

The `package-build` job installs the generated wheel into an isolated virtual environment and runs the CLI from that installed artifact. The `container-smoke` job builds the image and verifies both its CLI and package import. Tagged PyPI releases and GHCR publication call the reusable Python CI workflow and cannot publish unless all of these checks succeed for the tagged commit. GHCR publication is intentionally tag-only: ordinary pushes to `main` do not publish release images.
