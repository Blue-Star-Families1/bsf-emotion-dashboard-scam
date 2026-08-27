# bsf-emotion-dashboard
A Streamlit dashboard analyzing emotion trends in BSF forum discussions
![BSF AI app presentation](https://github.com/user-attachments/assets/147ce477-9171-4caa-8ff2-9ed8d66c2eff)

## Local run

1. Install dependencies from `requirements.txt`.
2. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`, replace every placeholder, and do **not** commit it.
3. Run the app with Streamlit and open the local URL.

For dev-container stability, `.streamlit/config.toml` sets `server.fileWatcherType = "none"` to prevent file-watcher mount hangs.

Project development guardrails are stored in `.github/copilot-instructions.md`. The repo is currently optimized for output quality, reliability, and performance improvements before any feature expansion.

## Runtime acceleration behavior

- CPU / GPU / MPS selection is based on what the runtime environment supports.
- It does **not** dynamically switch from CPU to GPU just because the laptop is busy or other apps are open.
- Dynamic device switching under load is intentionally avoided because it usually increases latency, model reload overhead, and instability.

## Streamlit Community Cloud deployment checklist

- Repository path is correct and accessible to the connected GitHub account.
- Branch is `main`.
- Main file path is `app.py`.
- Python version is set to `3.11` (aligned with `.python-version`).
- App secrets are configured in Streamlit Cloud **App settings → Secrets**.

If deploy logs show repeated clone failures (`🐙 Failed to download the sources`), verify:

- The repo exists at the exact owner/name.
- The Streamlit-connected GitHub identity has access to the repository.
- Organization/SAML/SSO authorization is granted for the repo.

## Secrets handling

- Never commit real credentials, API keys, private keys, or app passwords.
- Use `.streamlit/secrets.example.toml` as a template.
- Add a `[users]` section to secrets for login access, for example `"admin@example.com" = "change-me"`.
- Keep real values only in local ignored files or Streamlit Cloud Secrets UI.

## Production release gate

Before deploying to production:

- Run `test_tier1.py` and require **100% pass**.
- Run `test_pipeline.py` and require no policy-drift or integration errors.
- Snowflake-backed scripts are live integration tests. Set `RUN_LIVE_TESTS=1` only in an environment with working Snowflake credentials; requested live tests fail if their data is unavailable.
- Keep Tier-1 escalation policy consistent across code, UI text, and tests:
	- severity classification escalates scores `>= 5`, and shortlink evidence is always high severity.
- Verify secrets are present in Streamlit Cloud and rotated if any previous key exposure occurred.
- Gemini SDK policy:
	- Prefer `google-genai` (modern SDK) in production.
	- Keep `google-generativeai` only as a temporary compatibility fallback.
