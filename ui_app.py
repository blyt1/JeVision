"""Local UI: drop in a tile image, get the Qwen3-VL zero-shot defect score.

    .venv/bin/python ui_app.py   ->  http://127.0.0.1:7860
"""
import base64
import csv
import io

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel
from sklearn.metrics import roc_curve
from transformers import AutoModelForImageTextToText, AutoProcessor

from tile_readout_baseline import QUESTIONS, single_token_ids

MODEL, TILE = "Qwen/Qwen3-VL-4B-Instruct", 448
DEVICE, DTYPE = ("cuda", torch.bfloat16) if torch.cuda.is_available() else (
    ("mps", torch.float16) if torch.backends.mps.is_available() else ("cpu", torch.float32))

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEVICE).eval()
yes_ids = single_token_ids(processor.tokenizer, ["Yes", "yes"])
no_ids = single_token_ids(processor.tokenizer, ["No", "no"])


def flag_threshold(path="tile_scores.csv", fpr_target=0.05):
    """Margin giving ~5% FPR on the 2,000-tile baseline run (0 if the file is missing)."""
    try:
        rows = list(csv.DictReader(open(path)))
        fpr, _, thr = roc_curve([int(r["label"]) for r in rows], [float(r["margin"]) for r in rows])
        return float(thr[(fpr <= fpr_target).nonzero()[0].max()])
    except Exception:
        return 0.0


THRESHOLD = flag_threshold()
app = FastAPI()


class Req(BaseModel):
    image: str  # base64 (no data: prefix)
    question: str = "v1"


@app.post("/score")
def score(req: Req):
    img = Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB").resize((TILE, TILE))
    conv = [[{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": QUESTIONS[req.question]}]}]]
    inputs = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        last = model(**inputs).logits[:, -1, :].float()
    margin = (torch.logsumexp(last[:, yes_ids], 1) - torch.logsumexp(last[:, no_ids], 1)).item()
    return {"margin": margin, "score": torch.sigmoid(torch.tensor(margin)).item(),
            "threshold": THRESHOLD, "flagged": margin > THRESHOLD}


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Tile Defect Check</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--bd:#ccc;--ok:#1a7f37;--bad:#c62828}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mut:#999;--bd:#444;--ok:#4cc26a;--bad:#ef6b6b}}
body{background:var(--bg);color:var(--fg);font:16px system-ui;max-width:560px;margin:2rem auto;padding:0 16px}
#drop{border:2px dashed var(--bd);border-radius:12px;padding:2rem;text-align:center;color:var(--mut);cursor:pointer}
#drop.over{border-color:var(--fg)} img{max-width:100%;border-radius:8px;margin-top:1rem}
#res{margin-top:1rem;font-size:1.3rem;font-weight:600} small{display:block;color:var(--mut);font-weight:400;font-size:.85rem;margin-top:.3rem}
</style>
<h1>Tile defect check</h1>
<div id=drop>Drop a tile image here, or click to choose</div>
<input id=f type=file accept="image/*" hidden>
<img id=pv hidden><div id=res></div>
<script>
const $=id=>document.getElementById(id),drop=$('drop'),f=$('f');
drop.onclick=()=>f.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');go(e.dataTransfer.files[0])};
f.onchange=()=>go(f.files[0]);
async function go(file){
  if(!file)return;
  const url=URL.createObjectURL(file);$('pv').src=url;$('pv').hidden=false;
  $('res').textContent='Scoring…';
  const b64=await new Promise(r=>{const fr=new FileReader();fr.onload=()=>r(fr.result.split(',')[1]);fr.readAsDataURL(file)});
  try{
    const r=await fetch('/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:b64})});
    if(!r.ok)throw new Error(await r.text());
    const d=await r.json();
    $('res').innerHTML=`<span style="color:var(--${d.flagged?'bad':'ok'})">${d.flagged?'Possible defect':'Looks OK'}</span>`+
      `<small>Yes−No margin ${d.margin.toFixed(2)} (flag above ${d.threshold.toFixed(2)}, about 5% false alarms on test tiles). Zero-shot Qwen3-VL-4B; a ranking score, not a probability.</small>`;
  }catch(e){$('res').textContent='Error: '+e.message}
}
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    print(f"device={DEVICE} flag threshold={THRESHOLD:.2f}")
    uvicorn.run(app, host="127.0.0.1", port=7860)
