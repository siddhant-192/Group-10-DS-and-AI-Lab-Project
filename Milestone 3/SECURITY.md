# Security and credential handling

This repository does not contain API keys, OAuth tokens, service-account
credentials, rclone configuration, browser cookies, or Hugging Face tokens.

## Authentication

- Google Colab authentication is completed interactively by
  `scripts/setup_colab_cli.sh` and stored outside the repository.
- Public Hugging Face models normally require no token. If rate limiting
  requires one, authenticate through the Hugging Face CLI or an environment
  variable; never place the token in a script or JSON file.
- Google Drive/rclone is optional cold storage and is not part of the
  reproducible execution path.

## Before a public push

Run:

```bash
bash scripts/audit_public_package.sh
```

Also inspect staged files with `git diff --cached --stat` and
`git diff --cached`. Do not force-add files ignored by `.gitignore`.

If a credential is ever committed, revoke it immediately and remove it from
Git history; deleting it only in a later commit is insufficient.

