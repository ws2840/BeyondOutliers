# Beyond Outliers

## Environment Setup
```
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Risk Model
End-of-month (EOM) factor loadings are sourced from the [JKP factors](https://jkpfactors.com) dataset ([`Documentation.pdf`](https://data.jkpfactors.com/documents/Documentation.pdf)).

## Universe Membership
S&P 500 constituents are sourced from [`sp500.csv`](https://github.com/fja05680/sp500/blob/master/sp500.csv) in [fja05680/sp500](https://github.com/fja05680/sp500).