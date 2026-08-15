# Revision record

The diagnosis was moved into `hunyuan_groupnorm_diagnosis`.

The SSH instructions now use:

1. Plain `python`.
2. Paths relative to the repository root.
3. Shared source clones inside the diagnosis folder.
4. Logs, images, tensor dumps, environment details, patch output, and comparison output inside `hunyuan_groupnorm_diagnosis/artifacts`.

The diagnosis folder also contains reusable image comparison and tensor replay scripts. The live GroupNorm probe is stored as an applicable patch.

`run_diagnosis.sh` runs the complete workflow and creates a separate artifact directory for each invocation.
