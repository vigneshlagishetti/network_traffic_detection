# Ignore Guidelines and Recovery Commands

This file captures the steps taken to remove large artifacts from Git and keep them ignored in the future.

Use-case: the repository had large model files and processed datasets tracked (e.g. `models/*.joblib`, `data/processed/*.npz`). Pushing failed due to missing/corrupt Git LFS objects. The safe approach is to remove those files from the Git index and keep them out of future commits.

Quick commands used (run from repo root):

1) Create a backup branch (keep original pointers locally):

```bash
git branch backup-main
```

2) Remove large files from the index (keep local copies) and amend the commit:

```bash
git rm --cached models/*.joblib data/processed/*.npz -r
git commit --amend --no-edit
```

3) Push rewritten branch to origin:

```bash
git push -u origin main
```

Notes and recommendations
- If you need to keep large binary models in the repository, use Git LFS:

```bash
git lfs install
git lfs track "*.joblib"
git lfs track "data/processed/*.npz"
git add .gitattributes
git add models/*.joblib data/processed/*.npz
git commit -m "Add model and processed data to Git LFS"
git lfs push --all origin main
git push origin main
```

- If you do not want model files in Git history at all, keep them in external storage and add `models/` and `data/processed/` to `.gitignore` (this repo already has those entries).

- If you have backups of the original LFS object files, you can restore them into `.git/lfs/objects/<oid>` and then push. Otherwise, avoid pushing branches that reference missing LFS objects.

This guideline file is informational and intended to be tracked in the repository for future reference.
