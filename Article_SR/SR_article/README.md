# Article draft (LaTeX)

Черновик статьи, разбитый по темам в `sections/`.

## Build

```bash
cd Article_SR/SR_article
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Библиография: локальный файл `Bibliography.bib` (удобно заливать на Overleaf целиком).

## Structure

| File | Topic |
|------|--------|
| `main.tex` | Title, authors, `\input`, bibliography |
| `Bibliography.bib` | Local BibTeX database (for Overleaf) |
| `figures/` | Article PNGs (`fig01_…` … `fig06_…`) |
| `notebooks/make_article_figures.ipynb` | Build RD plots (`fig03`–`fig05`) from `runs/` |
| `sections/01_abstract.tex` | Abstract + keywords |
| `sections/02_introduction.tex` | Introduction |
| `sections/02b_related_work.tex` | Related work |
| `sections/03_problem.tex` | Problem formulation |
| `sections/04_method.tex` | Method |
| `sections/05_experiments.tex` | Experiments |
| `sections/06_conclusion.tex` | Conclusion |

## Figures

- `fig01`, `fig02`, `fig06` — put manually into `figures/`
- `fig03`–`fig05` — from your runs via the notebook (needs `runs/` on Zhores):

```bash
cd Article_SR/SR_article/notebooks
jupyter nbconvert --to notebook --execute make_article_figures.ipynb
# or open in Jupyter and Run All
```

