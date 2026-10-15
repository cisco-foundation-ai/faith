# Security Policies and Procedures

This document outlines security procedures and general policies for the
Foundation-AI Testing Hub `FAITH` project.

- [Disclosing a security issue](#disclosing-a-security-issue)
- [Vulnerability management](#vulnerability-management)
- [Running agentic benchmarks](#running-agentic-benchmarks)
- [Suggesting changes](#suggesting-changes)

## Disclosing a security issue

The `FAITH` maintainers take all security issues in the project
seriously. Thank you for improving the security of `FAITH`. We
appreciate your dedication to responsible disclosure and will make every effort
to acknowledge your contributions.

`FAITH` leverages GitHub's private vulnerability reporting.

To learn more about this feature and how to submit a vulnerability report,
review [GitHub's documentation on private reporting](https://docs.github.com/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability).

Here are some helpful details to include in your report:

- a detailed description of the issue
- the steps required to reproduce the issue
- versions of the project that may be affected by the issue
- if known, any mitigations for the issue

A maintainer will acknowledge the report within three (3) business days, and
will send a more detailed response within an additional three (3) business days
indicating the next steps in handling your report.

If you've been unable to successfully draft a vulnerability report via GitHub
or have not received a response during the allotted response window, please
reach out via the [Cisco Open security contact email](mailto:oss-security@cisco.com).

After the initial reply to your report, the maintainers will endeavor to keep
you informed of the progress towards a fix and full announcement, and may ask
for additional information or guidance.

## Vulnerability management

When the maintainers receive a disclosure report, they will assign it to a
primary handler.

This person will coordinate the fix and release process, which involves the
following steps:

- confirming the issue
- determining affected versions of the project
- auditing code to find any potential similar problems
- preparing fixes for all releases under maintenance

## Running agentic benchmarks

An agentic benchmark executes commands a model wrote. `FAITH` runs them in a
container or a Kubernetes sandbox, never on the host, but how strong that
boundary is depends on the backend and on the operator:

- The `docker` environment is meant for local runs of benchmark code you trust.
  Each task gets its own container with no network and no host paths mounted, a
  cap on the number of processes, all Linux capabilities dropped, and privilege
  escalation forbidden. A container still shares the host kernel, and the daemon
  that starts it is root-equivalent, so this bounds a runaway agent rather than
  containing a hostile one.
- The `kubernetes` environment claims a sandbox `FAITH` does not define.
  Isolation is whatever the SandboxTemplate and the cluster provide: the pod's
  runtime class, seccomp profile, NetworkPolicy and node pool are all the
  operator's to set. Kubernetes alone is not a hostile-code sandbox; untrusted
  workloads need an appropriately isolated runtime and verified cluster controls.

### Read-only benchmark inputs

Benchmarks may require `agentic_config.read_only_workspace: true`; both VulnLoc
phases do. FAITH protects workspace files and directories from the agent's shell,
disables subsequent uploads, and refuses to start the agent if it cannot establish
the requested policy. Scratch outside the workspace, such as `/tmp`, may remain
writable. This is an input-integrity guarantee, not a command allowlist, a network
policy, or protection from kernel/container escapes.

Docker requires a trusted, non-root image without added Linux capabilities. The
Kubernetes runtime must implement the irreversible seal, retain policy state
across sidecar restarts, and execute agents without privilege escalation. The
bundled template includes these controls; custom deployments must supply them.
Startup checks verify basic behavior, not the security of an entire cluster.

FAITH maintainers own the implementation, shipped defaults and reference code.
Operators own image provenance, host/cluster patching, RBAC, network enforcement,
secrets, runtime isolation and deployment verification. The bundled privileged
sidecar is part of the trusted computing base, not an additional isolation boundary.

Treat model output and benchmark data as untrusted input. A task whose prompt or
data steers the agent at the host, the network or your credentials will be
followed. Run agentic benchmarks on a machine you are willing to lose, holding
no credentials the run does not need.

`FAITH` does not provide or operate the container host, the cluster, or the
network they sit on -- the operator does, and the isolation those provide is the
operator's to configure and verify. `FAITH` is provided without warranty of any
kind (see the [LICENSE](LICENSE)); choosing a boundary appropriate to the code
under evaluation is the operator's responsibility.

## Suggesting changes

If you have suggestions on how this process could be improved please submit an
issue or pull request.
