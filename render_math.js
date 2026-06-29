// 将 paper.md 中的 LaTeX 公式渲染为 SVG，输出 formulas.json
const fs = require('fs');
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages });
const svg = new SVG({ fontCache: 'local' });
const doc = mathjax.document('', { InputJax: tex, OutputJax: svg });

const src = fs.readFileSync('docs/paper.md', 'utf8');
const out = {};
function render(t, display) {
  const node = doc.convert(t, { display });
  return adaptor.innerHTML(node);
}
// display $$...$$
src.replace(/\$\$([^$]+)\$\$/g, (m, t) => { out[m] = render(t.trim(), true); return m; });
// inline $...$
src.replace(/\$([^$\n]+)\$/g, (m, t) => { if(!out[m]) out[m] = render(t.trim(), false); return m; });
fs.writeFileSync('formulas.json', JSON.stringify(out));
console.log('rendered', Object.keys(out).length, 'formulas');
