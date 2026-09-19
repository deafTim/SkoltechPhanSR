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
| `sections/01_abstract.tex` | Abstract + keywords |
| `sections/02_introduction.tex` | Introduction (motivation, question, approach, contributions) |
| `sections/02b_related_work.tex` | Related work + positioning table |
| `sections/03_problem.tex` | Problem formulation |
| `sections/04_method.tex` | Method: reformulation, ADMM, projection, loss, algorithm, setup |
| `sections/05_experiments.tex` | Experiments (placeholder) |
| `sections/06_conclusion.tex` | Conclusion (placeholder) |
