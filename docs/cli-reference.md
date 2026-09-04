# CLI reference

Install the project before using the `careercast` executable:

```bash
python -m pip install -e .
careercast --help
```

The API defaults to `http://127.0.0.1:8000`. Set `CAREERCAST_API_URL` or place
`--api-url URL` before the subcommand to override it.

## Commands

### health

```bash
careercast health
```

Checks service and model availability.

### models

```bash
careercast models
```

Displays embedding, classifier, class-count, and metrics information.

### predict

```bash
careercast predict "Python, SQL, pandas" --top-k 5
```

Returns Logistic Regression predictions. `--top-k` defaults to 5.

### recommend

```bash
careercast recommend "Python, SQL, machine learning" --top-k 10
```

Returns weighted ensemble recommendations. `--top-k` defaults to 10.

### gap

```bash
careercast gap "Python, SQL" --target-career "Data Scientist" --top-k-careers 1
```

If `--target-career` is omitted, the API automatically selects target careers
from its ensemble ranking. `--top-k-careers` defaults to 5.

## Output and exit status

Commands print formatted JSON to standard output. A successful command returns
exit code `0`. Connection and API failures print a JSON `error` object and
return exit code `1`.
