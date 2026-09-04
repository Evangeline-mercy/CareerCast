# Public release checklist

## Code and tests

- [ ] Working tree is clean.
- [ ] Focused local test suite passes.
- [ ] GitHub Actions passes on the release commit.
- [ ] Wheel and source distribution build successfully.
- [ ] Package installs in a clean environment and `careercast --help` works.

## Documentation

- [ ] README contains the public application link and local setup instructions.
- [ ] API and CLI references match current command and request schemas.
- [ ] Dataset and model cards report recorded evidence and limitations.
- [ ] Deployment guide matches current Streamlit configuration.

## Application

- [ ] Public URL loads and reports `API connected`.
- [ ] Individual prediction, recommendation, gap analysis, and PDF export work.
- [ ] Cohort analytics and CSV export work.
- [ ] Career comparison chart, table, and missing-skill lists work.
- [ ] PDF is visually checked for wrapping, margins, and page breaks.

## Release record

- [ ] Update the final version if necessary.
- [ ] Tag the verified commit (planned Milestone 4 tag: `v4.0.0`).
- [ ] Publish release notes summarizing package, tests, UI, and documentation.
- [ ] Include the repository URL, deployed application URL, CI evidence, and
      selected screenshots in the mentor submission.
