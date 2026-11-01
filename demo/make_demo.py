"""Build ltr_demo.html from ltr_demo_template.html, ntcoef.js and ltrlim.js.

The template contains the page (markup, styles, user interface); the
computational core (the JavaScript port of the LTRATIO programs: the single
integral DFX_CORE, the double integral TR_LIMDIST with the closed-form start
TR_NSSTART, TR_CFLIM, TR_JBSTART and the moments TR_LIMMOM) is kept in
ltrlim.js, with the series coefficients of NT_SERIES in ntcoef.js.
Both are pasted into the template at the __CORE__ marker, inside a
<script type="text/plain"> element from which the page creates its Web
Workers.  Embedding them keeps the demo a single file that runs from disk; a
worker cannot load a separate script from a file:// location.

    python make_demo.py            (writes ltr_demo.html in the same folder)
"""
import os
here = os.path.dirname(os.path.abspath(__file__))
read = lambda f: open(os.path.join(here, f), encoding='utf-8').read()
core = read('ntcoef.js') + '\n' + read('ltrlim.js')
tpl = read('ltr_demo_template.html')
body = tpl.replace('__CORE__', core)
i = body.index('</style>') + len('</style>')
html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + body[:i] + '\n</head>\n<body>' + body[i:] + '\n</body>\n</html>\n')
out = os.path.join(here, 'ltr_demo.html')
open(out, 'w', encoding='utf-8').write(html)
print('wrote', out, len(html), 'bytes')
