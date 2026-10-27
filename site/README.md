# Project page

Static GitHub Pages site for Terrazzo: `index.html` (English) and `zh.html` (中文),
sharing `styles.css`, `app.js` and `favicon.svg`. No build step, no external JS/CSS
dependencies.

Deployment is `.github/workflows/pages.yml`, which runs on changes under `site/`:
it validates the pages with `scripts/check_site.py`, uploads the folder as a Pages
artifact, and deploys it. The site is published at
<https://louisyeaaah.github.io/terrazzo/>.

To preview locally:

```bash
python3 -m http.server 8000 --directory site
# open http://127.0.0.1:8000/
```

When benchmark numbers change, update the README/BENCHMARKS first, then the numbers
and tables here (they are intentionally duplicated as a snapshot; the page links back
to `BENCHMARKS.md` for the full methodology).
