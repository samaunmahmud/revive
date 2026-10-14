#!/usr/bin/env bash
# Render docs/REPORT.md to docs/ReVive-report.pdf.
# Markdown -> HTML with GitHub's renderer (gh api /markdown, same as the repo page),
# then a print stylesheet, then headless Chrome prints it. Needs: gh (logged in), Chrome.
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
BODY=$(gh api /markdown -f mode=gfm -F text=@REPORT.md)
cat > .report.html <<HTML
<!doctype html><html><head><meta charset="utf-8"><title>ReVive technical report</title>
<base href="file://$PWD/">
<style>
  @page { size: A4; margin: 16mm 15mm 18mm; }
  body { font: 10.2pt/1.5 -apple-system, "Segoe UI", Inter, Helvetica, Arial, sans-serif; color: #1d1915; max-width: 100%; }
  h1 { font: italic 700 24pt/1.15 Georgia, "Iowan Old Style", serif; margin: 0 0 6pt; }
  h2 { font: 700 14pt/1.3 Georgia, serif; border-bottom: 1px solid #e0d6c4; padding-bottom: 3pt; margin: 18pt 0 6pt; break-after: avoid; }
  h3 { font-size: 11.2pt; margin: 13pt 0 4pt; break-after: avoid; }
  p, li { orphans: 3; widows: 3; }
  a { color: #8a5214; text-decoration: none; }
  code { font: 8.8pt Menlo, Consolas, monospace; background: #f4efe6; padding: 0 3px; border-radius: 3px; }
  pre { background: #f6f2ea; padding: 8pt; border-radius: 6px; font-size: 8.6pt; white-space: pre-wrap; break-inside: avoid; }
  pre code { background: none; padding: 0; }
  table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; break-inside: avoid; width: 100%; }
  th, td { border: 1px solid #e3dccd; padding: 3pt 6pt; text-align: left; vertical-align: top; }
  th { background: #f6f1e7; }
  img { max-width: 100%; max-height: 95mm; width: auto; break-inside: avoid; display: block; margin: 6pt auto; }
  img[src$=".svg"], img[src*="real-photos"] { max-height: 150mm; }
  blockquote { margin: 8pt 0; padding: 4pt 10pt; border-left: 3px solid #e2a04f; background: #fbf6ec; color: #4a443c; }
  hr { border: 0; border-top: 1px solid #e0d6c4; }
  .markdown-heading a.anchor { display: none; }
</style></head><body>
$BODY
</body></html>
HTML
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/ReVive-report.pdf" "file://$PWD/.report.html" 2>/dev/null
rm -f .report.html
echo "wrote docs/ReVive-report.pdf"
