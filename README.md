# Beyond Outliers

## Environment Setup
```
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Risk Model
End-of-month (EOM) factor loadings are sourced from the [JKP factors](https://jkpfactors.com) dataset ([`Documentation.pdf`](https://data.jkpfactors.com/documents/Documentation.pdf)).

### Cross-sectional Regression

$$
R_i = \alpha + \sum_{j=1}^J \delta_{i \in j} \beta_j + \sum_{k=1}^K l_{ki} f_k + \varepsilon
$$

- $R_i$: raw return of the $i^{th}$ stock
- $\delta_{i \in j}$: industry exposure of the $i^{th}$ stock to the $j^{th}$ of $J$ industries, $1$ if $i \in j$ and $0$ otherwise.
    - Each stock belongs to exactly one industry
- $\beta_j$: market return of the $j^{th}$ industry
- $l_{ki}$: loading of the $i^{th}$ stock on the $k^{th}$ of $K$ factors
- $f_k$: factor return of the $k^{th}$ factor

## Universe Membership
S&P 500 constituents are sourced from [`sp500.csv`](https://github.com/fja05680/sp500/blob/master/sp500.csv) in [fja05680/sp500](https://github.com/fja05680/sp500).