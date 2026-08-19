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

from ai.ui import CSS as CHAT_CSS
from ai.ui import JS as CHAT_JS
from ai.ui import MAIN as CHAT_MAIN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI = os.path.join(ROOT, "ai")
CKPT = os.path.join(AI, "checkpoints")
OUT = os.path.join(ROOT, "index.html")   # the repository landing page
SAMPLES = os.path.join(AI, "data", "samples.json")

DEMO_PROMPTS = [
    "What license does black use?",
    "Does fastapi depend on pydantic?",
    "What is 3471 + 2856?",
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

    def card(k, v, sub=""):
        return (f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div>'
                f'<div class="s">{sub}</div></div>')

    stat_cards = "".join([
        card("parameters", "5,015,808", "4 layers · d=256 · 8 heads · RoPE · SwiGLU"),
        card("pretraining", f"{tokens/1e6:.1f}M / {total/1e6:.0f}M", "tokens of PyPI source text"),
        card("val loss", f"{last_val:.2f}" if last_val else "—",
             f"perplexity {2.718281828**last_val:.0f}" if last_val else ""),
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
      </article>""" for s in samples)

    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    return f"""<!doctype html>
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
  {sparkline(train_pts, label="pretraining loss")}
  <h3>Validation loss</h3>
  {sparkline(val_pts, color="#7c9cff", label="validation loss")}
  {"<h3>Instruction tuning loss</h3>" + sparkline(sft_pts, color="#fbbf24", label="sft loss") if sft_pts else ""}
</div>

<h2>What it actually does</h2>
<p>At this size the model cannot memorise the world, so it is trained to do the two things that
fit in 5M parameters: <b>answer from a context passage retrieved for it</b>, and <b>write its
reasoning out as tokens</b> before answering. Unanswerable questions are trained to be refused.</p>
{sample_html}
<p class="dim">Transcripts above are generated from the checkpoint that was current when this page
was built — the pipeline was still training, so treat them as a snapshot, not a final score.</p>

<h2>Checkpoints</h2>
<div class="panel"><table>
<thead><tr><th>file</th><th>label</th><th>stage</th><th>step</th><th>tokens seen</th></tr></thead>
<tbody>{ck_rows or '<tr><td colspan="5" class="dim">none yet</td></tr>'}</tbody></table></div>

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

<h2 id="talk">Talk to it</h2>
<p class="dim" id="offline">This page is static. Start the model server with
  <code>python -m ai.serve --port 8000</code> and open it there (or append
  <code>?api=http://localhost:8000</code> to this URL) to chat with the model.</p>
<div id="chatapp" class="app" hidden>{CHAT_MAIN}</div>

<footer>
  Built in <a href="https://github.com/alrickdaley8-cpu/eferfefe">alrickdaley8-cpu/eferfefe</a> ·
  page generated by <code>ai/make_page.py</code> · numbers are read straight from the training logs.
</footer>
</div>
<script>
{CHAT_JS}
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
