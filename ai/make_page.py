"""Generate ai/index.html — a static, dependency-free page about the tiny-lm project.

It bakes in the real numbers (loss curves from the training logs, checkpoint list, dataset sizes)
and a handful of real transcripts, so the page works on GitHub Pages with no server. If the chat
server happens to be running on the same host, the page unlocks a live demo panel automatically.

    python ai/make_page.py                # regenerate index.html with fresh numbers + samples
    python ai/make_page.py --no-samples   # keep the transcripts already on the page
"""
from __future__ import annotations

import argparse
import html
import json
import os
import time

import hashlib

from ai.ui import CSS as CHAT_CSS
from ai.ui import JS as CHAT_JS
from ai.ui import MAIN as CHAT_MAIN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI = os.path.join(ROOT, "ai")
CKPT = os.path.join(AI, "checkpoints")
OUT = os.path.join(ROOT, "index.html")   # the repository landing page
SAMPLES = os.path.join(AI, "data", "samples.json")

DEMO_PROMPTS = [
    "What can you do?",
    "What license does black use?",
    "Does fastapi depend on pydantic?",
    "What is 3471 + 2856?",
    "How do I read a text file in Python?",
    "Who won the 2038 World Cup?",
]


# ----------------------------------------------------------------------------- data
def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def ui_hash() -> str:
    """Fingerprint of the embedded chat client — lets a test catch a stale index.html."""
    blob = (CHAT_CSS + CHAT_MAIN + CHAT_JS).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def page_data(max_points: int = 120) -> dict:
    """Everything the page shows, as JSON — served live by ai.serve at /page-data."""
    d = collect()

    def thin(points):
        if len(points) <= max_points:
            return points
        step = len(points) / max_points
        return [points[int(i * step)] for i in range(max_points)]

    samples = []
    if os.path.exists(SAMPLES):
        try:
            samples = json.load(open(SAMPLES))
        except Exception:
            samples = []
    st = d["status"]
    tokens = st.get("tokens", 0)
    val = [r["val_loss"] for r in d["pre"] if "val_loss" in r]
    return {
        "curves": {
            "pretrain": thin([[r["tok"], r["ema"]] for r in d["pre"] if "ema" in r and "tok" in r]),
            "val": thin([[r["step"] * 8192, r["val_loss"]] for r in d["pre"] if "val_loss" in r]),
            "sft": thin([[r["tok"], r["ema"]] for r in d["sft"] if "ema" in r and "tok" in r]),
        },
        "status": st,
        "kb": d["kb"],
        "checkpoints": d["ckpts"],
        "samples": samples,
        "stats": {"tokens": tokens, "total_tokens": st.get("total_tokens", 100_000_000),
                  "val_loss": val[-1] if val else None, "stage": st.get("stage"),
                  "step": st.get("step", 0)},
        "generated_at": time.time(),
    }


def collect() -> dict:
    pre = read_jsonl(os.path.join(CKPT, "log.jsonl"))
    sft = read_jsonl(os.path.join(CKPT, "sft_log.jsonl"))
    status = {}
    try:
        status = json.load(open(os.path.join(CKPT, "status.json")))
    except Exception:
        pass
    kb = 0
    kb_path = os.path.join(AI, "data", "qa", "knowledge.jsonl")
    if os.path.exists(kb_path):
        with open(kb_path) as f:
            kb = sum(1 for _ in f)
    ckpts = []
    try:
        from ai.chat import list_checkpoints
        ckpts = list_checkpoints()
    except Exception:
        pass
    return {"pre": pre, "sft": sft, "status": status, "kb": kb, "ckpts": ckpts}


def gather_samples(limit_tokens: int = 90) -> list[dict]:
    from ai.chat import Assistant
    a = Assistant()
    out = []
    for q in DEMO_PROMPTS:
        r = a.answer(q, max_tokens=limit_tokens, temperature=0.2, top_k=5)
        out.append({"q": q, "reasoning": r.get("reasoning", ""), "a": r.get("answer", ""),
                    "context": (r.get("context") or "")[:220],
                    "verification": r.get("verification", "unchecked"),
                    "checkpoint": r.get("checkpoint"), "step": r.get("step")})
        print(f"[page] sampled: {q}", flush=True)
    return out


# ----------------------------------------------------------------------------- chart
def sparkline(points: list[tuple[float, float]], w=680, h=210, pad=34,
              color="#5eead4", label="") -> str:
    """Minimal inline SVG line chart — no JS, no libraries."""
    if len(points) < 2:
        return '<p class="dim">not enough data yet</p>'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    y0, y1 = y0 - (y1 - y0) * 0.08, y1 + (y1 - y0) * 0.08
    sx = lambda x: pad + (x - x0) / max(1e-9, x1 - x0) * (w - pad * 1.4)      # noqa: E731
    sy = lambda y: h - pad - (y - y0) / max(1e-9, y1 - y0) * (h - pad * 1.7)  # noqa: E731
    d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                 for i, (x, y) in enumerate(points))
    area = d + f" L{sx(xs[-1]):.1f},{h-pad:.1f} L{sx(xs[0]):.1f},{h-pad:.1f} Z"
    grid = "".join(
        f'<line x1="{pad}" y1="{sy(y):.1f}" x2="{w-pad*0.4:.0f}" y2="{sy(y):.1f}" '
        f'class="grid"/><text x="{pad-8}" y="{sy(y)+4:.1f}" class="ylab">{y:.1f}</text>'
        for y in [y0 + (y1 - y0) * f for f in (0.05, 0.35, 0.65, 0.95)])
    return f"""<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{html.escape(label)}">
  <defs><linearGradient id="g{color[1:]}" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0%" stop-color="{color}" stop-opacity=".28"/>
    <stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>
  {grid}
  <path d="{area}" fill="url(#g{color[1:]})"/>
  <path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
  <text x="{pad}" y="{h-8}" class="xlab">{xs[0]/1e6:.1f}M tokens</text>
  <text x="{w-pad*1.4:.0f}" y="{h-8}" class="xlab" text-anchor="end">{xs[-1]/1e6:.1f}M</text>
</svg>"""


# ----------------------------------------------------------------------------- page
def render(d: dict, samples: list[dict]) -> str:
    pre, sft, st, kb = d["pre"], d["sft"], d["status"], d["kb"]
    train_pts = [(r["tok"], r["ema"]) for r in pre if "ema" in r and "tok" in r]
    val_pts = [(r["step"] * 8192, r["val_loss"]) for r in pre if "val_loss" in r]
    sft_pts = [(r["tok"], r["ema"]) for r in sft if "ema" in r and "tok" in r]

    tokens = st.get("tokens", train_pts[-1][0] if train_pts else 0)
    total = st.get("total_tokens", 100_000_000)
    last_val = val_pts[-1][1] if val_pts else None

    def card(k, v, sub="", cid=""):
        idattr = f' id="card-{cid}"' if cid else ""
        return (f'<div class="stat"><div class="k">{k}</div><div class="v"{idattr}>{v}</div>'
                f'<div class="s">{sub}</div></div>')

    stat_cards = "".join([
        card("parameters", "5,015,808", "4 layers · d=256 · 8 heads · RoPE · SwiGLU"),
        card("pretraining", f"{tokens/1e6:.1f}M / {total/1e6:.0f}M", "tokens of PyPI source text",
             cid="tokens"),
        card("val loss", f"{last_val:.2f}" if last_val else "—",
             f"perplexity {2.718281828**last_val:.0f}" if last_val else "", cid="val"),
        card("knowledge base", f"{kb:,}", "packages available to retrieval"),
        card("instruction data", "288k examples", "146k QA + 142.5k chain-of-thought"),
        card("hardware", "2 vCPU", "no GPU, fp32, ~4,000 tok/s"),
    ])

    ck_rows = "".join(
        f"<tr><td class='mono'>{html.escape(c['file'])}</td><td>{html.escape(c['label'])}</td>"
        f"<td>{c['stage']}</td><td class='num'>{c['step']:,}</td>"
        f"<td class='num'>{c['tokens']/1e6:.1f}M</td></tr>" for c in d["ckpts"])

    sample_html = "".join(f"""
      <article class="turn">
        <div class="q">{html.escape(s['q'])}</div>
        {f'<div class="ctx"><span>retrieved</span>{html.escape(s["context"])}…</div>' if s.get("context") else ''}
        {f'<div class="think"><span>thinking</span>{html.escape(s["reasoning"])}</div>' if s.get("reasoning") else ''}
        <div class="a">{html.escape(s['a']) or '<em>(empty)</em>'}</div>
        <div class="vb {s.get('verification','unchecked')}">{
          {"ok": "verified against the knowledge base",
           "corrected": "corrected from the knowledge base",
           "fallback": "model output rejected as noise — fallback reply",
           "unchecked": "unverified — raw model output, nothing to check it against"
           }.get(s.get('verification','unchecked'), '')}</div>
      </article>""" for s in samples)

    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    build = ui_hash()
    return f"""<!doctype html>
<!-- tiny-lm page · ui-build {build} · generated {ts} · regenerate with: python ai/make_page.py -->
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tiny-lm — a 5M parameter language model trained from scratch</title>
<meta name="description" content="A 5,015,808 parameter transformer pretrained on 100M tokens and
instruction tuned for retrieval-grounded reasoning, on 2 CPU cores.">
<style>
:root{{--bg:#0b0e13;--panel:#12161f;--panel2:#161b26;--line:#242c3a;--text:#e8eef7;--dim:#8b98ad;
 --accent:#5eead4;--accent2:#7c9cff}}
*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(1100px 600px at 12% -10%,#16203055,transparent),var(--bg);
 color:var(--text);font:15px/1.7 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.mono,code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.wrap{{max-width:1000px;margin:0 auto;padding:54px 20px 80px}}
h1{{font-size:38px;line-height:1.15;margin:0 0 10px;letter-spacing:-.5px}}
h2{{font-size:20px;margin:44px 0 14px;letter-spacing:-.2px}}
h3{{font-size:15px;margin:22px 0 8px;color:var(--accent)}}
p{{color:#cbd5e6}} .dim{{color:var(--dim)}}
.lede{{font-size:17px;color:#b9c6da;max-width:70ch}}
.badges{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 4px}}
.badge{{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:5px 12px;
 font-size:12.5px;color:var(--dim)}}
.badge b{{color:var(--text);font-weight:600}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:22px 0}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px}}
.stat .k{{font-size:11.5px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim)}}
.stat .v{{font-size:23px;font-weight:650;margin:3px 0 1px}}
.stat .s{{font-size:12.5px;color:var(--dim)}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 20px}}
.chart{{width:100%;height:auto;display:block}}
line.grid{{stroke:#1e2634;stroke-width:1}}
text.ylab{{fill:#5f6d82;font:10px ui-monospace,monospace;text-anchor:end}}
text.xlab{{fill:#5f6d82;font:10px ui-monospace,monospace}}
table{{width:100%;border-collapse:collapse;font-size:13.5px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}
th{{color:var(--dim);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.5px}}
td.num{{text-align:right;font-family:ui-monospace,monospace}}
pre{{background:#0d121a;border:1px solid var(--line);border-radius:12px;padding:14px 16px;
 overflow:auto;font-size:13px;color:#cfe0f5}}
.turn{{border:1px solid var(--line);background:var(--panel);border-radius:14px;padding:14px 16px;
 margin:12px 0}}
.turn .q{{font-weight:600}}
.turn .ctx,.turn .think{{font-size:12.8px;border-left:2px solid var(--line);padding:6px 0 6px 11px;
 margin:9px 0;color:var(--dim);white-space:pre-wrap}}
.turn .think{{border-left-color:var(--accent2);color:#a8b6cc}}
.turn .ctx span,.turn .think span{{display:block;font-size:10.5px;text-transform:uppercase;
 letter-spacing:.6px;color:var(--accent2);margin-bottom:3px}}
.turn .a{{margin-top:8px;white-space:pre-wrap}}
.turn .vb{{margin-top:9px;font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;
 display:inline-block;border:1px solid;border-radius:999px;padding:2px 9px}}
.turn .vb.ok{{color:var(--accent);border-color:#5eead444}}
.turn .vb.corrected{{color:#fbbf24;border-color:#fbbf2444}}
.turn .vb.unchecked{{color:var(--dim);border-color:#8b98ad44}}
.turn .vb.fallback{{color:#f87171;border-color:#f8717144}}
.flow{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.flowstep{{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:11px 13px;
 font-size:13px}}
.flowstep b{{display:block;color:var(--accent);font-size:11.5px;text-transform:uppercase;
 letter-spacing:.5px;margin-bottom:3px}}
a{{color:var(--accent2)}} footer{{margin-top:52px;color:var(--dim);font-size:12.5px}}
</style>
<style>{CHAT_CSS}</style>
<style>
/* the chat widget is embedded in the page flow rather than being its own screen */
#chatapp.app{{max-width:none;margin:0;padding:0;grid-template-columns:1fr 330px}}
#chatapp .chat{{min-height:360px}}
#chatapp #log{{max-height:400px}}
#chatapp .side{{position:static}}
@media(max-width:980px){{#chatapp.app{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">

<h1>tiny-lm</h1>
<p class="lede">A <b>5,015,808 parameter</b> language model built from scratch — corpus collection,
tokenizer, transformer, pretraining on <b>100,000,000 tokens</b>, instruction tuning and
retrieval-grounded chain-of-thought reasoning — all trained on <b>2 CPU cores</b>, no GPU.</p>

<div class="badges">
  <span class="badge"><b>4</b> layers</span>
  <span class="badge"><b>256</b> d_model</span>
  <span class="badge"><b>8192</b> vocab</span>
  <span class="badge"><b>512</b> context</span>
  <span class="badge">RoPE · SwiGLU · RMSNorm · tied embeddings</span>
  <span class="badge">updated {ts}</span>
</div>

<div class="cards">{stat_cards}</div>

<h2>Pipeline</h2>
<div class="flow">
  <div class="flowstep"><b>1 · corpus</b>470 MB of text from 1,745 PyPI source releases, deduped and
    quality filtered</div>
  <div class="flowstep"><b>2 · tokenizer</b>byte-level BPE, 8,192 merges, trained on the corpus</div>
  <div class="flowstep"><b>3 · pretrain</b>100M tokens, 1 epoch, cosine LR, 12,207 steps</div>
  <div class="flowstep"><b>4 · instruct</b>146k QA pairs from 11,922 package records</div>
  <div class="flowstep"><b>5 · reason</b>142.5k worked-step examples, answer-only loss</div>
  <div class="flowstep"><b>6 · serve</b>BM25 retrieval + streaming chat with a visible thought process</div>
</div>

<h2>Training curves</h2>
<div class="panel">
  <h3>Pretraining loss (EMA)</h3>
  <div id="curve-pretrain">{sparkline(train_pts, label="pretraining loss")}</div>
  <h3>Validation loss</h3>
  <div id="curve-val">{sparkline(val_pts, color="#7c9cff", label="validation loss")}</div>
  <div id="curve-sft">{"<h3>Instruction tuning loss</h3>" + sparkline(sft_pts, color="#fbbf24", label="sft loss") if sft_pts else ""}</div>
</div>

<h2>What it actually does</h2>
<p>At this size the model cannot memorise the world, so it is trained to do the two things that
fit in 5M parameters: <b>answer from a context passage retrieved for it</b>, and <b>write its
reasoning out as tokens</b> before answering. Unanswerable questions are trained to be refused.</p>
<div id="samples">{sample_html}</div>
<p class="dim">Transcripts above are generated from the checkpoint that was current when this page
was built — the pipeline was still training, so treat them as a snapshot, not a final score.</p>

<h2>Checkpoints</h2>
<div class="panel"><table>
<thead><tr><th>file</th><th>label</th><th>stage</th><th>step</th><th>tokens seen</th></tr></thead>
<tbody id="ckbody">{ck_rows or '<tr><td colspan="5" class="dim">none yet</td></tr>'}</tbody>
</table></div>

<h2>Run it yourself</h2>
<pre>git clone https://github.com/alrickdaley8-cpu/eferfefe
cd eferfefe
python -m venv .venv &amp;&amp; .venv/bin/pip install torch numpy tokenizers requests

python ai/build_corpus.py --target-mb 450     # 470 MB of text
python ai/train_tokenizer.py                  # 8k byte-level BPE
python ai/prepare_data.py                     # exactly 100,000,000 tokens
python ai/build_qa.py --packages 12000        # knowledge base + QA set
python ai/build_reasoning.py                  # chain-of-thought set

ai/daemon.sh start                            # pretrain -> instruction tune, in the background
ai/daemon.sh status                           # progress, loss, throughput, ETA
python -m ai.chat --think "does fastapi depend on pydantic?"
python -m ai.serve --port 8000                # streaming chat UI</pre>

<p class="dim" id="freshness">snapshot baked in at build time</p>

<h2 id="talk">Talk to it</h2>
<p class="dim" id="offline">Looking for a model server… if none is running, start one with
  <code>python -m ai.serve --port 8000</code> and reload, or point this page at a server with
  <code>?api=https://host:8000</code>.</p>
<p class="dim" id="apinote" style="font-size:12.5px"></p>
<div id="chatapp" class="app" hidden>{CHAT_MAIN}</div>

<footer>
  Built in <a href="https://github.com/alrickdaley8-cpu/eferfefe">alrickdaley8-cpu/eferfefe</a> ·
  page generated by <code>ai/make_page.py</code> · numbers are read straight from the training logs.
</footer>
</div>
<script>window.TINYLM_MANUAL_INIT=true;</script>
<script>
{CHAT_JS}
</script>
<script>
// Redraw the baked-in snapshot from live data whenever a server is reachable, so the page is
// current no matter when it was generated.
function spark(points,color){{
  if(!points||points.length<2) return '<p class="dim">not enough data yet</p>';
  const w=680,h=210,pad=34;
  const xs=points.map(p=>p[0]), ys=points.map(p=>p[1]);
  const x0=Math.min(...xs),x1=Math.max(...xs);
  let y0=Math.min(...ys),y1=Math.max(...ys);
  const span=(y1-y0)||1; y0-=span*0.08; y1+=span*0.08;
  const sx=x=>pad+(x-x0)/((x1-x0)||1)*(w-pad*1.4);
  const sy=y=>h-pad-(y-y0)/((y1-y0)||1)*(h-pad*1.7);
  const d=points.map((p,i)=>`${{i?'L':'M'}}${{sx(p[0]).toFixed(1)}},${{sy(p[1]).toFixed(1)}}`).join(' ');
  const area=`${{d}} L${{sx(xs[xs.length-1]).toFixed(1)}},${{h-pad}} L${{sx(xs[0]).toFixed(1)}},${{h-pad}} Z`;
  const grid=[0.05,0.35,0.65,0.95].map(f=>{{
    const y=y0+(y1-y0)*f;
    return `<line x1="${{pad}}" y1="${{sy(y).toFixed(1)}}" x2="${{w-pad*0.4}}" y2="${{sy(y).toFixed(1)}}" class="grid"/>`
         + `<text x="${{pad-8}}" y="${{(sy(y)+4).toFixed(1)}}" class="ylab">${{y.toFixed(1)}}</text>`;
  }}).join('');
  const gid='g'+color.slice(1);
  return `<svg viewBox="0 0 ${{w}} ${{h}}" class="chart"><defs>
    <linearGradient id="${{gid}}" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0%" stop-color="${{color}}" stop-opacity=".28"/>
    <stop offset="100%" stop-color="${{color}}" stop-opacity="0"/></linearGradient></defs>
    ${{grid}}<path d="${{area}}" fill="url(#${{gid}})"/>
    <path d="${{d}}" fill="none" stroke="${{color}}" stroke-width="2" stroke-linejoin="round"/>
    <text x="${{pad}}" y="${{h-8}}" class="xlab">${{(xs[0]/1e6).toFixed(1)}}M tokens</text>
    <text x="${{w-pad*1.4}}" y="${{h-8}}" class="xlab" text-anchor="end">${{(xs[xs.length-1]/1e6).toFixed(1)}}M</text></svg>`;
}}
const VB={{ok:"verified against the knowledge base",
          corrected:"corrected from the knowledge base",
          fallback:"model output rejected as noise — fallback reply",
          unchecked:"unverified — raw model output, nothing to check it against"}};
async function refreshPage(){{
  let d; try{{ d=await (await fetch(API+"/page-data")).json(); }}catch(e){{ return; }}
  const st=d.stats||{{}};
  const set=(id,v)=>{{const el=document.getElementById(id); if(el&&v!=null) el.textContent=v;}};
  set("card-tokens",`${{(st.tokens/1e6).toFixed(1)}}M / ${{(st.total_tokens/1e6).toFixed(0)}}M`);
  set("card-val", st.val_loss!=null ? st.val_loss.toFixed(2) : "—");
  const curves=d.curves||{{}};
  const cp=document.getElementById("curve-pretrain"); if(cp) cp.innerHTML=spark(curves.pretrain,"#5eead4");
  const cv=document.getElementById("curve-val"); if(cv) cv.innerHTML=spark(curves.val,"#7c9cff");
  const cs=document.getElementById("curve-sft");
  if(cs && curves.sft && curves.sft.length>1)
    cs.innerHTML="<h3>Instruction tuning loss</h3>"+spark(curves.sft,"#fbbf24");
  const tb=document.getElementById("ckbody");
  if(tb && d.checkpoints && d.checkpoints.length)
    tb.innerHTML=d.checkpoints.map(c=>`<tr><td class="mono">${{c.file}}</td><td>${{c.label}}</td>`+
      `<td>${{c.stage}}</td><td class="num">${{c.step.toLocaleString()}}</td>`+
      `<td class="num">${{(c.tokens/1e6).toFixed(1)}}M</td></tr>`).join("");
  const sec=document.getElementById("samples");
  if(sec && d.samples && d.samples.length)
    sec.innerHTML=d.samples.map(s=>`<article class="turn"><div class="q">${{s.q}}</div>`+
      (s.context?`<div class="ctx"><span>retrieved</span>${{s.context}}…</div>`:"")+
      (s.reasoning?`<div class="think"><span>thinking</span>${{s.reasoning}}</div>`:"")+
      `<div class="a">${{s.a||"(empty)"}}</div>`+
      `<div class="vb ${{s.verification||"unchecked"}}">${{VB[s.verification||"unchecked"]}}</div>`+
      `</article>`).join("");
  const f=document.getElementById("freshness");
  if(f) f.textContent="live — figures refreshed from the running server "+
    new Date().toLocaleTimeString();
}}
initChat(found=>{{
  if(found){{ refreshPage(); setInterval(refreshPage,30000); }}
  document.getElementById("chatapp").hidden=!found;
  document.getElementById("offline").hidden=found;
  if(found && API) document.getElementById("apinote").textContent="connected to "+(API||"this page");
}});
</script>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-samples", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    d = collect()
    samples: list[dict] = []
    if os.path.exists(SAMPLES):
        samples = json.load(open(SAMPLES))
    if not args.no_samples:
        try:
            samples = gather_samples()
            os.makedirs(os.path.dirname(SAMPLES), exist_ok=True)
            json.dump(samples, open(SAMPLES, "w"), indent=1)
        except Exception as exc:
            print(f"[page] sampling skipped: {exc}", flush=True)

    with open(args.out, "w") as f:
        f.write(render(d, samples))
    print(f"[page] wrote {args.out} ({os.path.getsize(args.out)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
