Vendor provenance and notice
===========================

This directory contains vendored code that was copied from the upstream
"mlx-lm" project to make the toolkit self-contained for development and
testing. The primary vendored tree is under:

  core/vendor/mlx_lm/models/

Source
------

The files were copied from the upstream repository:

  https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models

Date of copy
------------

5 October 2025

License & attribution
---------------------

The vendored files are third-party code. Before redistribution or
publicly releasing a derivative work, verify the upstream license terms
and comply with any notice or attribution requirements specified by the
upstream project. The upstream repository contains licensing information
in its root; consult that repo for the authoritative license text.

Recommended actions
-------------------

- Review the upstream license and add any required LICENSE/NOTICE files to
  this repository if you plan to distribute the vendored code.
- Consider recording the upstream commit hash (or tag) used for the copy
  for traceability. You can add this next to the Date of copy above.
- Keep vendor changes isolated and clearly documented. When updating the
  vendored tree, make a single commit with a clear message such as
  "vendor: update mlx_lm/models from ml-explore/mlx-lm @ <commit>".

Why we vendor
-------------

Vendoring is useful for:
- Ensuring reproducible development without depending on an external
  package manager or upstream packaging.
- Allowing small, local modifications or shims that ease integration with
  this toolkit (for example, providing a lightweight registry or tests).

If you'd prefer not to vendor, consider using a git submodule or adding
the upstream project as a formal dependency.

Contact
-------

If you want me to: (A) commit this file and stage the vendor files, (B)
generate a short NOTICE with an upstream commit hash, or (C) revert any
particular vendor files, tell me which and I'll execute it.
