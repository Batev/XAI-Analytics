# Research experiments

This directory preserves local research separately from the application in the
repository root and `util/`. Archived code is exploratory; it is not an additional
supported application entry point or an automated test suite.

## Local work archived on 31 October 2026

See [the archive notes](local-work-2026-10-31/README.md) and
[the file inventory](local-work-2026-10-31/inventory.json).

The research concerns LIME explanation stability and kernel-width optimization,
agreement between feature rankings using rank-biased overlap (RBO), and plotting
experiments on the existing Income and Cervical datasets.

Notebook source cells and attachments are preserved. Saved outputs, execution
counts, execution timing, and notebook widget state were cleared for review.
Historical helpers and snippets are preserved without implementation changes.
Their original imports and assumptions may require adaptation before execution.

## Application baseline

During curation, the following files were restored to commit
`f68b9b8998123cab521041bc05dc8e0d3d05b3cb`:

- `util/commons.py`
- `requirements.txt`
- `requirements-no-deps.txt`
- `XAI-Analytics.ipynb`
- `XAI-Example-Income.ipynb`

This removes the incomplete multiprocessing edit that prevented parsing, the
hard-coded SHAP output class, fixed experimental training parameters and seed,
and notebook debugging edits. It also restores the committed dependency inputs.
It preserves the committed LIME stability optimizer already in the project.

Restoring this baseline does not modernize dependencies or resolve other issues
in the historical application. The older
[project assessment](../docs/project-direction-assessment-2026-10-24.md)
documents broader work separately.

## Local originals and generated files

An ignored, verified ZIP at
`.local-backups/2026-10-31-before-curation.zip` contains 147 original files,
including all five modified application files, the research files, saved
notebook results, exports, the downloaded Skater source and archive, IDE state,
checkpoints, logs, and utility bytecode. `BACKUP-MANIFEST.json` inside the ZIP
records original paths, sizes and SHA-256 hashes. This local backup is not part
of Git; retain it separately if the original outputs are needed elsewhere.

Downloaded Skater files, exported results and the log were moved to the ignored
`.local-artifacts/` directory. IDE state, checkpoints and utility caches remain
in place and are ignored. The downloaded Skater directory no longer shadows an
installed package at the repository root. No existing work was deleted.

## Validation performed

- Verified original bytes against the ZIP manifest before restoring files.
- Verified every archived notebook retains its original cell types, sources
  and attachments, and every curated file matches its recorded SHA-256 hash.
- Verified all five restored application files match the base commit exactly.
- Parsed all application Python modules and imported `util.commons`,
  `util.model`, `util.dataset`, `util.split` and `util.ui` using the existing
  Python 3.9 virtual environment.

No model training, notebook execution, fresh installation or dependency upgrade
was performed during this curation.
