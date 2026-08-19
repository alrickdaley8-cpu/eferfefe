"""The single-page UI served by ai/serve.py (kept separate so the server stays readable)."""

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tiny-lm · 5M parameter assistant</title>
<style>
:root{
  --bg:#0b0e13; --panel:#12161f; --panel2:#161b26; --line:#242c3a; --text:#e8eef7;
  --dim:#8b98ad; --accent:#5eead4; --accent2:#7c9cff; --warn:#fbbf24; --user:#1d2735;
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 700px at 15% -10%,#16203055,transparent),var(--bg);
  color:var(--text);font:14.5px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
code,pre,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.app{max-width:1240px;margin:0 auto;padding:22px 20px 40px;display:grid;grid-template-columns:1fr 380px;
  gap:20px;align-items:start}
header{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:14px;align-items:center;justify-content:space-between;
  padding-bottom:16px;border-bottom:1px solid var(--line);margin-bottom:4px}
.brand{display:flex;align-items:baseline;gap:10px}
.brand h1{font-size:19px;margin:0;letter-spacing:.2px}
.brand .tag{color:var(--dim);font-size:12.5px}
.pill{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:5px 11px;
  font-size:12px;color:var(--dim);display:inline-flex;align-items:center;gap:7px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent)}
.dot.off{background:#64748b;box-shadow:none}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px}
.chat{padding:6px 6px 0;min-height:420px;display:flex;flex-direction:column}
#log{flex:1;overflow-y:auto;max-height:min(60vh,620px);padding:14px 14px 4px;scroll-behavior:smooth}
.msg{display:flex;gap:11px;margin:0 0 16px}
.avatar{flex:0 0 28px;height:28px;border-radius:8px;display:grid;place-items:center;font-size:12px;
  background:var(--panel2);border:1px solid var(--line);color:var(--dim)}
.msg.you .avatar{background:var(--user);color:var(--accent2)}
.bubble{flex:1;min-width:0}
.who{font-size:11.5px;color:var(--dim);letter-spacing:.4px;text-transform:uppercase;margin-bottom:3px}
.body{white-space:pre-wrap;word-wrap:break-word}
.body.mono{font-size:13.5px}
.cursor{display:inline-block;width:7px;height:15px;background:var(--accent);vertical-align:-2px;
  animation:blink 1s steps(2) infinite}
@keyframes blink{50%{opacity:0}}
.composer{display:flex;gap:10px;padding:12px 14px;border-top:1px solid var(--line);background:var(--panel2);
  border-radius:0 0 14px 14px}
textarea#q{flex:1;resize:none;height:44px;max-height:130px;background:var(--bg);color:var(--text);
  border:1px solid var(--line);border-radius:10px;padding:11px 12px;font:inherit;outline:none}
textarea#q:focus{border-color:var(--accent2)}
button{background:linear-gradient(180deg,#2dd4bf,#14b8a6);color:#04201c;border:0;border-radius:10px;
  padding:0 18px;font:600 14px/1 inherit;cursor:pointer}
button:disabled{opacity:.45;cursor:not-allowed}
button.ghost{background:transparent;color:var(--dim);border:1px solid var(--line);padding:6px 11px;
  font-weight:500;font-size:12.5px}
.chips{display:flex;flex-wrap:wrap;gap:7px;padding:10px 14px 0}
.chip{border:1px solid var(--line);background:var(--panel2);border-radius:999px;padding:5px 11px;
  font-size:12px;color:var(--dim);cursor:pointer}
.chip:hover{color:var(--text);border-color:var(--accent2)}
.side{display:flex;flex-direction:column;gap:14px;position:sticky;top:22px}
.sec{padding:13px 15px}
.sec h2{font-size:12px;letter-spacing:.7px;text-transform:uppercase;color:var(--dim);margin:0 0 10px;
  display:flex;justify-content:space-between;align-items:center}
select,input[type=range]{width:100%}
select{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:9px;
  padding:9px 10px;font:inherit;outline:none}
.mdesc{color:var(--dim);font-size:12px;margin-top:7px}
.ctrl{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:9px 0;font-size:13px}
.ctrl input[type=range]{width:135px}
.switch{position:relative;width:38px;height:21px;background:var(--line);border-radius:99px;cursor:pointer;
  transition:.15s;flex:0 0 auto}
.switch.on{background:#0d9488}
.switch::after{content:"";position:absolute;top:2px;left:2px;width:17px;height:17px;border-radius:50%;
  background:#e2e8f0;transition:.15s}
.switch.on::after{left:19px}
#think{max-height:min(52vh,520px);overflow-y:auto;font-size:12.6px}
.step{border-left:2px solid var(--line);padding:2px 0 10px 11px;margin-left:3px;position:relative}
.step:last-child{padding-bottom:0}
.step .k{color:var(--accent);font-size:11px;letter-spacing:.5px;text-transform:uppercase}
.step .v{color:var(--dim);white-space:pre-wrap;word-break:break-word;margin-top:2px}
.step.active .k{color:var(--warn)}
.cand{display:flex;justify-content:space-between;gap:8px;padding:3px 0;border-bottom:1px dashed #202838}
.cand .n{color:var(--text)}.cand .s{color:var(--accent2)}
.bars{display:flex;flex-direction:column;gap:4px;margin-top:5px}
.bar{display:grid;grid-template-columns:78px 1fr 38px;align-items:center;gap:7px}
.bar .t{color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:pre}
.bar .g{height:7px;background:var(--panel2);border-radius:99px;overflow:hidden}
.bar .g i{display:block;height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent))}
.bar .p{color:var(--dim);text-align:right}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:7px 12px;font-size:12.4px;color:var(--dim)}
.stats b{color:var(--text);font-weight:600}
.prog{height:6px;background:var(--panel2);border-radius:99px;overflow:hidden;margin:8px 0 6px}
.prog i{display:block;height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent))}
details summary{cursor:pointer;color:var(--dim);font-size:12px}
@media(max-width:980px){.app{grid-template-columns:1fr}.side{position:static}}
</style></head>
<body><div class="app">

<header>
  <div class="brand">
    <h1>tiny-lm</h1>
    <span class="tag" id="brandTag">5,015,808 parameters · trained from scratch</span>
  </div>
  <div style="display:flex;gap:9px;flex-wrap:wrap">
    <span class="pill"><span class="dot" id="trainDot"></span><span id="trainPill">training…</span></span>
    <span class="pill" id="kbPill">knowledge base —</span>
  </div>
</header>

<main class="card chat">
  <div id="log"></div>
  <div class="chips" id="chips"></div>
  <div class="composer">
    <textarea id="q" placeholder="Ask about a Python package…  (Enter to send, Shift+Enter for newline)"></textarea>
    <button id="go">Send</button>
  </div>
</main>

<aside class="side">
  <section class="card sec">
    <h2>Model <button class="ghost" id="refresh">refresh</button></h2>
    <select id="model"></select>
    <div class="mdesc" id="mdesc">—</div>
  </section>

  <section class="card sec">
    <h2>Thinking</h2>
    <div id="think"><div class="step"><div class="v">Ask something to see the retrieval and
      decoding trace.</div></div></div>
  </section>

  <section class="card sec">
    <h2>Decoding</h2>
    <div class="ctrl"><span>Grounding (RAG)</span><div class="switch on" id="rag"></div></div>
    <div class="ctrl"><span>Chat template</span><div class="switch on" id="tpl"></div></div>
    <div class="ctrl"><span>Temperature</span><input type="range" id="temp" min="10" max="140" value="70"></div>
    <div class="ctrl"><span class="mono" id="tempV">0.70</span><span class="mono" id="topkV">top-k 40</span></div>
    <div class="ctrl"><span>Top-k</span><input type="range" id="topk" min="1" max="200" value="40"></div>
    <div class="ctrl"><span>Max tokens</span><input type="range" id="maxt" min="20" max="320" value="160"></div>
    <div class="ctrl"><span class="mono" id="maxtV">160 tokens</span></div>
  </section>

  <section class="card sec">
    <h2>Training</h2>
    <div class="prog"><i id="progBar" style="width:0%"></i></div>
    <div class="stats" id="tstats"></div>
  </section>
</aside>
</div>

<script>
const $=id=>document.getElementById(id);
const EXAMPLES=["What is beautifulsoup4?","How do I install black?","What license does requests use?",
 "Which package should I use for progress bars?","What are the dependencies of fastapi?",
 "How do I read a text file in Python?","What is 128 + 46?","Who won the 2038 World Cup?"];
let busy=false, opts={rag:true,tpl:true};

$("chips").innerHTML=EXAMPLES.map(e=>`<span class="chip">${esc(e)}</span>`).join("");
$("chips").onclick=e=>{if(e.target.classList.contains("chip")){$("q").value=e.target.textContent;send();}};
["rag","tpl"].forEach(k=>$(k).onclick=()=>{opts[k]=!opts[k];$(k).classList.toggle("on",opts[k]);});
$("temp").oninput=e=>$("tempV").textContent=(e.target.value/100).toFixed(2);
$("topk").oninput=e=>$("topkV").textContent="top-k "+e.target.value;
$("maxt").oninput=e=>$("maxtV").textContent=e.target.value+" tokens";
$("go").onclick=send;
$("q").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}});
$("refresh").onclick=loadModels;
$("model").onchange=()=>{const o=$("model").selectedOptions[0];$("mdesc").textContent=o.dataset.desc||"";};

function esc(s){return (s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}

function addMsg(who,cls){
  const d=document.createElement("div"); d.className="msg "+cls;
  d.innerHTML=`<div class="avatar">${cls==="you"?"you":"lm"}</div>
    <div class="bubble"><div class="who">${who}</div><div class="body"></div></div>`;
  $("log").appendChild(d); $("log").scrollTop=$("log").scrollHeight;
  return d.querySelector(".body");
}
function thought(kind,html,active){
  const d=document.createElement("div"); d.className="step"+(active?" active":"");
  d.innerHTML=`<div class="k">${esc(kind)}</div><div class="v">${html}</div>`;
  $("think").appendChild(d); $("think").scrollTop=$("think").scrollHeight; return d;
}
function clearThoughts(){$("think").innerHTML="";}

async function loadModels(){
  const d=await (await fetch("/models")).json();
  $("model").innerHTML=d.models.map(m=>
    `<option value="${m.file}" data-desc="${esc(m.description)} · step ${m.step.toLocaleString()} · ${(m.tokens/1e6).toFixed(1)}M tokens seen"
      ${m.loaded?"selected":""}>${esc(m.label)} — ${m.stage}</option>`).join("");
  $("mdesc").textContent=$("model").selectedOptions[0]?.dataset.desc||"";
  $("brandTag").textContent=`${d.params.toLocaleString()} parameters · vocab ${d.vocab} · ${d.block_size}-token context`;
  $("kbPill").textContent=`knowledge base ${d.kb.toLocaleString()} packages`;
}
async function loadStatus(){
  try{
    const s=await (await fetch("/status")).json();
    const frac=(s.tokens||0)/(s.total_tokens||1);
    $("progBar").style.width=(frac*100).toFixed(1)+"%";
    $("trainDot").className="dot"+(s.live?"":" off");
    $("trainPill").textContent=s.live?`${s.stage} · ${(frac*100).toFixed(1)}%`:"training idle";
    $("tstats").innerHTML=[
      ["stage",s.stage||"—"],["state",s.state||"—"],
      ["tokens",`${((s.tokens||0)/1e6).toFixed(2)}M / ${((s.total_tokens||0)/1e6).toFixed(0)}M`],
      ["step",`${(s.step||0).toLocaleString()} / ${(s.total_steps||0).toLocaleString()}`],
      ["loss",s.loss??"—"],["tok/s",s.tok_per_s?Math.round(s.tok_per_s):"—"],
      ["eta",s.eta_s?fmt(s.eta_s):"—"],["best val",s.best_val??"—"],
    ].map(([k,v])=>`<span>${k}</span><b>${esc(String(v))}</b>`).join("");
  }catch(e){}
}
function fmt(s){s=Math.round(s);const h=Math.floor(s/3600),m=Math.round((s%3600)/60);return h?`${h}h${String(m).padStart(2,"0")}m`:`${m}m`;}

async function send(){
  if(busy) return;
  const text=$("q").value.trim(); if(!text) return;
  $("q").value=""; busy=true; $("go").disabled=true;
  addMsg("you","you").textContent=text;
  clearThoughts();
  const body=addMsg("tiny-lm","bot"); body.innerHTML='<span class="cursor"></span>';
  let out="", live=null;

  const res=await fetch("/chat/stream",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message:text,model:$("model").value,use_context:opts.rag,
      chat_template:opts.tpl,temperature:$("temp").value/100,top_k:+$("topk").value,
      tokens:+$("maxt").value})});
  const reader=res.body.getReader(), dec=new TextDecoder(); let buf="";
  while(true){
    const {value,done}=await reader.read(); if(done) break;
    buf+=dec.decode(value,{stream:true});
    const parts=buf.split("\n\n"); buf=parts.pop();
    for(const p of parts){
      if(!p.startsWith("data: ")) continue;
      const ev=JSON.parse(p.slice(6));
      if(ev.type==="thought"){
        if(ev.kind==="candidates"){
          thought("retrieved candidates",ev.candidates.map(c=>
            `<div class="cand"><span class="n">${esc(c.name)}</span><span class="s">${c.score}</span></div>
             <div style="color:#6b7a90">${esc(c.summary)}</div>`).join(""));
        }else if(ev.kind==="prompt"){
          thought("prompt sent to the model",
            `<details><summary>${ev.tokens} tokens in, ${ev.budget} max out</summary>
             <div class="mono" style="margin-top:6px">${esc(ev.text)}</div></details>`);
        }else{
          thought(ev.kind,esc(ev.text));
        }
      }else if(ev.type==="token"){
        out+=ev.text;
        body.innerHTML=esc(out)+'<span class="cursor"></span>';
        $("log").scrollTop=$("log").scrollHeight;
        if(!live) live=thought("decoding","",true);
        live.querySelector(".v").innerHTML=
          `<div style="color:#8b98ad">token ${ev.i+1} · confidence ${(ev.confidence*100).toFixed(0)}%</div>
           <div class="bars">`+ev.alternatives.map(a=>
            `<div class="bar"><span class="t mono">${esc(JSON.stringify(a.token).slice(1,-1))}</span>
             <span class="g"><i style="width:${(a.p*100).toFixed(1)}%"></i></span>
             <span class="p">${(a.p*100).toFixed(0)}%</span></div>`).join("")+`</div>`;
      }else if(ev.type==="done"){
        body.innerHTML=esc(ev.answer||"(empty answer)");
        if(live){live.classList.remove("active");}
        const s=ev.stats;
        thought("done",`${s.generated_tokens} tokens · ${s.tok_per_s} tok/s · ${s.total_s}s total
          · prompt ${s.prompt_tokens} tokens`);
      }else if(ev.type==="error"){
        body.innerHTML=`<span style="color:#f87171">${esc(ev.text)}</span>`;
      }
    }
  }
  busy=false; $("go").disabled=false; $("q").focus();
}

loadModels(); loadStatus(); setInterval(loadStatus,15000); setInterval(loadModels,120000);
</script></body></html>
"""
