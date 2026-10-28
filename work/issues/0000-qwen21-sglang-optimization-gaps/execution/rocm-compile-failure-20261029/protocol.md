# Saved graph storage diagnostic

Explore the GPU fault from the v1 compatibility stage-profile attempt. Use the
saved production Inductor graph from cdae0d00d58f44329e219c9b1ae4e46e7896d64c.
The graph's example inputs preserve actual shapes and strides but use synthetic
values. No checkpoint, full request, timing or model quality measurement is needed.

Hypothesis: functionalization of AITER's conservatively mutating Q/K/V schemas
introduces copies with storage sizes inconsistent with their read counts.
Inspect generated source and guard the two identified linear copies before
launch. Synchronize previous launches to establish that they completed. Stop at
the first unsafe copy. Do not launch a copy known to read outside its storage.

Acceptance: GPU health before and after, unchanged installed schemas recorded,
linear-copy source verified, original example geometry verified, and actual
input storage smaller than the generated read count. This identifies a concrete
invalid memory access in generated code, without claiming a full model fix.
Budget: one saved block graph invocation, 240 seconds supervised maximum.

Do not change installed libraries or model source. Push a separate main-based
experiment branch. Leave raw output remote and record local findings. Restore
remote main and verify no GPU process remains.
