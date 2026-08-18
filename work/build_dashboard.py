import html
import json
import math
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "outputs" / "fraud_detection.db"
MODEL = ROOT / "outputs" / "fraud_model.json"
OUT = ROOT / "outputs" / "painel-transacoes-suspeitas.html"
FRAGMENT = ROOT / "outputs" / "transacoes-suspeitas-fragment.html"
model = json.loads(MODEL.read_text(encoding="utf-8"))
features = model["features"]

with sqlite3.connect(DB) as con:
    con.row_factory = sqlite3.Row
    rows = con.execute(f"""
        SELECT a.transaction_id AS id, a.fraud_probability AS risk,
               a.actual_class AS actual, {', '.join('t.' + name for name in features)}
        FROM alerts a JOIN transactions t ON t.rowid = a.transaction_id
        ORDER BY a.fraud_probability DESC, a.transaction_id
    """).fetchall()

def technical_evidence(row):
    contributions = []
    normal, fraud = model["classes"]["0"], model["classes"]["1"]
    for index, name in enumerate(features):
        value = float(row[name])
        m0, v0 = normal["mean"][index], normal["var"][index]
        m1, v1 = fraud["mean"][index], fraud["var"][index]
        log0 = -0.5 * (math.log(2 * math.pi * v0) + ((value - m0) ** 2) / v0)
        log1 = -0.5 * (math.log(2 * math.pi * v1) + ((value - m1) ** 2) / v1)
        contributions.append((log1 - log0, name))
    strongest = sorted(contributions, reverse=True)[:3]
    signals = ", ".join(f"{name} ({score:+.1f})" for score, name in strongest)
    if row["actual"] == 1:
        return f"Confirmada pelo rótulo Class=1 do dataset. Maiores sinais técnicos do modelo: {signals}."
    return f"Não confirmada no dataset (Class=0). O alerta foi provocado principalmente por: {signals}."

data = [{"id": r["id"], "r": round(r["risk"] * 100, 4), "a": r["actual"],
         "t": round(r["Time"], 2), "v": round(r["Amount"], 2),
         "j": technical_evidence(r)} for r in rows]
summary = {
    "alerts": len(data),
    "confirmed": sum(x["a"] == 1 for x in data),
    "threshold": round(model["threshold"] * 100, 2),
    "recall": round(model["evaluation"]["recall"] * 100, 1),
}
artifacts = [
    ("Banco de transações e alertas", "fraud_detection.db", DB.stat().st_size),
    ("Modelo treinado e métricas", "fraud_model.json", MODEL.stat().st_size),
    ("Painel interativo", "painel-transacoes-suspeitas.html", 0),
]

def size_label(size):
    return f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024 else f"{size / 1024:.1f} KB"

fragment = f'''<div id="fraud-locator">
  <h1>Localizador de transações suspeitas</h1>
  <div class="viz-grid">
    <section class="card viz-stat"><span class="text-muted">Alertas</span><strong class="viz-stat-value">{summary['alerts']:,}</strong></section>
    <section class="card viz-stat"><span class="text-muted">Fraudes confirmadas</span><strong class="viz-stat-value">{summary['confirmed']:,}</strong></section>
    <section class="card viz-stat"><span class="text-muted">Recall no teste</span><strong class="viz-stat-value">{summary['recall']}%</strong></section>
  </div>
  <div class="viz-controls">
    <label class="form-label">Buscar ID <input id="fraud-search" class="form-control" type="search" inputmode="numeric" placeholder="Ex.: 541"></label>
    <label class="form-label">Risco mínimo: <span id="risk-value">{summary['threshold']}%</span><input id="risk-min" class="form-range" type="range" min="0" max="100" step="1" value="{int(summary['threshold'])}"></label>
    <label class="form-label">Situação <select id="fraud-class" class="form-select"><option value="all">Todas</option><option value="1">Fraude confirmada</option><option value="0">Falso positivo</option></select></label>
  </div>
  <p id="fraud-count" class="text-muted text-small" aria-live="polite"></p>
  <div class="table-responsive"><table class="table table-sm">
    <thead><tr><th>ID</th><th class="text-end">Risco</th><th class="text-end">Valor</th><th class="text-end">Tempo (s)</th><th>Situação conhecida</th><th>Justificativa</th></tr></thead>
    <tbody id="fraud-body"></tbody>
  </table></div>
</div>
<style>
#fraud-locator .viz-grid{{margin-bottom:1rem}} #fraud-locator .viz-controls{{margin:1rem 0}}
#fraud-locator .confirmed{{color:var(--destructive)}} #fraud-locator .risk{{font-weight:500}}
</style>
<script>
(()=>{{
const root=document.getElementById('fraud-locator');
const data={json.dumps(data, separators=(',', ':'))};
const search=root.querySelector('#fraud-search'), range=root.querySelector('#risk-min'), cls=root.querySelector('#fraud-class');
const body=root.querySelector('#fraud-body'), count=root.querySelector('#fraud-count'), riskValue=root.querySelector('#risk-value');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function render(){{
 const q=search.value.trim(), min=Number(range.value), c=cls.value;
 const filtered=data.filter(x=>(!q||String(x.id).includes(q))&&x.r>=min&&(c==='all'||String(x.a)===c));
 riskValue.textContent=min+'%'; count.textContent=filtered.length.toLocaleString('pt-BR')+' transações encontradas; exibindo até 100.';
 body.innerHTML=filtered.slice(0,100).map(x=>`<tr><td>${{esc(x.id)}}</td><td class="text-end risk">${{x.r.toLocaleString('pt-BR')}}%</td><td class="text-end">${{x.v.toLocaleString('pt-BR',{{style:'currency',currency:'EUR'}})}}</td><td class="text-end">${{x.t.toLocaleString('pt-BR')}}</td><td class="${{x.a===1?'confirmed':''}}">${{x.a===1?'Fraude confirmada':'Falso positivo'}}</td><td>${{esc(x.j)}}</td></tr>`).join('');
}}
[search,range,cls].forEach(el=>el.addEventListener('input',render)); render();
}})();
</script>'''

standalone = f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transações suspeitas</title><style>
:root{{--bg:#f6f7fb;--surface:#fff;--text:#172033;--muted:#667085;--border:#dfe3ea;--accent:#3157d5;--danger:#b42318}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:28px}}h1{{font-size:24px}}h2{{font-size:18px;margin-top:28px}}.stats,.files{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.stat,.file{{background:var(--surface);padding:18px;border:1px solid var(--border);border-radius:10px}}.stat strong{{display:block;font-size:26px;margin-top:6px}}.file strong,.file span{{display:block;margin-bottom:7px}}.file a{{color:var(--accent)}}.controls{{display:flex;gap:16px;align-items:end;flex-wrap:wrap;margin:22px 0}}label{{display:grid;gap:6px}}input,select{{font:inherit;padding:9px;border:1px solid var(--border);border-radius:7px;background:var(--surface)}}input[type=range]{{padding:0}}.table-wrap{{overflow:auto;background:var(--surface);border:1px solid var(--border);border-radius:10px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 12px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}}th{{position:sticky;top:0;background:var(--surface)}}.num{{text-align:right}}.confirmed{{color:var(--danger);font-weight:600}}.muted{{color:var(--muted)}}.tools{{margin-top:18px;color:var(--muted)}}@media(max-width:650px){{.stats,.files{{grid-template-columns:1fr}}main{{padding:16px}}}}</style></head><body><main>
<h1>Localizador de transações suspeitas</h1><div class="stats"><div class="stat"><span>Alertas</span><strong>{summary['alerts']:,}</strong></div><div class="stat"><span>Fraudes confirmadas</span><strong>{summary['confirmed']:,}</strong></div><div class="stat"><span>Recall no teste</span><strong>{summary['recall']}%</strong></div></div>
<div class="controls"><label>Buscar ID<input id="q" type="search" inputmode="numeric" placeholder="Ex.: 541"></label><label>Risco mínimo <output id="rv">{summary['threshold']}%</output><input id="r" type="range" min="0" max="100" value="{int(summary['threshold'])}"></label><label>Situação<select id="c"><option value="all">Todas</option><option value="1">Fraude confirmada</option><option value="0">Falso positivo</option></select></label></div><p id="count" class="muted"></p>
<div class="table-wrap"><table><thead><tr><th>ID</th><th class="num">Risco</th><th class="num">Valor</th><th class="num">Tempo (s)</th><th>Situação conhecida</th><th>Justificativa</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Arquivos conectados</h2><div class="files">{''.join(f'<div class="file"><strong>{html.escape(label)}</strong><span class="muted">{html.escape(name)} · {size_label(size) if size else "gerado neste painel"}</span><a href="{html.escape(name)}">Abrir arquivo</a></div>' for label, name, size in artifacts)}</div>
<p class="tools">Banco: tabela <code>alerts</code> ligada à tabela <code>transactions</code>.</p></main><script>
const data={json.dumps(data, separators=(',', ':'))},q=document.querySelector('#q'),r=document.querySelector('#r'),c=document.querySelector('#c'),rows=document.querySelector('#rows'),count=document.querySelector('#count'),rv=document.querySelector('#rv');
function draw(){{const f=data.filter(x=>(!q.value||String(x.id).includes(q.value))&&x.r>=+r.value&&(c.value==='all'||String(x.a)===c.value));rv.value=r.value+'%';count.textContent=f.length.toLocaleString('pt-BR')+' transações encontradas; exibindo até 250.';rows.innerHTML=f.slice(0,250).map(x=>`<tr><td>${{x.id}}</td><td class="num">${{x.r.toLocaleString('pt-BR')}}%</td><td class="num">${{x.v.toLocaleString('pt-BR',{{style:'currency',currency:'EUR'}})}}</td><td class="num">${{x.t.toLocaleString('pt-BR')}}</td><td class="${{x.a?'confirmed':''}}">${{x.a?'Fraude confirmada':'Falso positivo'}}</td><td>${{x.j}}</td></tr>`).join('')}}[q,r,c].forEach(x=>x.addEventListener('input',draw));draw();</script></body></html>'''

OUT.write_text(standalone, encoding="utf-8")
FRAGMENT.write_text(fragment, encoding="utf-8")
print(json.dumps({"dashboard": str(OUT), "visualization": str(FRAGMENT), "rows": len(data)}))

