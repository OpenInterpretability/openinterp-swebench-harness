#!/usr/bin/env python3
"""Control for the 'late' claim (late_channel.tex #1): is the late patch effect CONSOLIDATION or
patch-DILUTION? Logit-lens the decision-token residual per layer in the run-CoT and measure the margin
logP(ans_cot) - logP(ans_nocot). If the margin is ~0 early and only turns positive late -> the deciding
information consolidates late (real). If positive early -> info is present early and the early-patch failure
is dilution (artifact). No SAE, no patch. Run: colab run --gpu G4 --timeout 3600 logit_lens_control.py"""
import os, time, json
import torch, numpy as np, random as _r
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_ID=os.environ.get("MODEL","Qwen/Qwen3.6-27B"); THINK_MAX=512
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

PROBLEMS = [
 {"q":"Compute 23*17 + 14*9.","correct":"517","distractors":["505","391","523"]},
 {"q":"Compute 47*8 - 19*6.","correct":"262","distractors":["248","276","290"]},
 {"q":"Compute 12*12 - 11*13.","correct":"1","distractors":["0","2","3"]},
 {"q":"Sum of the integers 1 through 30.","correct":"465","distractors":["450","480","435"]},
 {"q":"Compute 7! / 5!.","correct":"42","distractors":["49","35","210"]},
 {"q":"Compute 2^10 - 3^6.","correct":"295","distractors":["305","285","256"]},
 {"q":"Compute gcd(84, 126).","correct":"42","distractors":["21","14","63"]},
 {"q":"Compute 15% of 240 plus 25% of 160.","correct":"76","distractors":["80","72","100"]},
 {"q":"Compute 17^2 - 13^2.","correct":"120","distractors":["116","130","108"]},
 {"q":"Compute sqrt(144) + sqrt(169).","correct":"25","distractors":["24","26","23"]},
 {"q":"Compute 100 - 7*8 - 6*4.","correct":"20","distractors":["24","16","28"]},
 {"q":"Compute 3*4*5*6 / (2*9).","correct":"20","distractors":["18","24","15"]},
 {"q":"Convert binary 11010 to decimal.","correct":"26","distractors":["24","28","22"]},
 {"q":"Compute 0xFF + 1 in decimal.","correct":"256","distractors":["255","254","257"]},
 {"q":"How many prime numbers are below 20?","correct":"8","distractors":["7","9","6"]},
 {"q":"Compute 5^3 + 4^3 + 3^3.","correct":"216","distractors":["206","226","196"]},
 {"q":"Compute 13*7 + 9*11 - 4*5.","correct":"170","distractors":["180","160","150"]},
 {"q":"Compute (2+3)^2 - 2^2 - 3^2.","correct":"12","distractors":["13","11","10"]},
 {"q":"Compute LCM(6, 8, 12).","correct":"24","distractors":["48","12","36"]},
 {"q":"How many positive divisors does 36 have?","correct":"9","distractors":["8","6","12"]},
 {"q":"Monty Hall: 3 doors, you pick one, the host opens a losing door, you switch. P(win)?","correct":"2/3","distractors":["1/2","1/3","3/4"]},
 {"q":"Flip a fair coin 3 times. P(at least one head)?","correct":"7/8","distractors":["1/2","3/4","5/8"]},
 {"q":"5 people each shake hands once with every other. Total handshakes?","correct":"10","distractors":["20","15","25"]},
 {"q":"At 3:15 on an analog clock, the angle between the hour and minute hands?","correct":"7.5 degrees","distractors":["0 degrees","30 degrees","22.5 degrees"]},
 {"q":"Roll two fair dice. P(sum = 7)?","correct":"1/6","distractors":["1/8","5/36","1/12"]},
 {"q":"Compute 8! / (6! * 2!).","correct":"28","distractors":["56","14","16"]},
 {"q":"In Python, len(set([1,1,2,3,3,3])) gives?","correct":"3","distractors":["6","4","2"]},
 {"q":"In Python, 'abcdef'[::2] gives?","correct":"'ace'","distractors":["'bdf'","'abc'","'adf'"]},
 {"q":"In Python, 10 // 3 + 10 % 3 gives?","correct":"4","distractors":["3","5","7"]},
 {"q":"In Python, bin(10).count('1') gives?","correct":"2","distractors":["1","3","4"]},
 {"q":"In Python, len(list(range(2, 20, 3))) gives?","correct":"6","distractors":["5","7","18"]},
 {"q":"How many trailing zeros are in 25! (25 factorial)?","correct":"6","distractors":["5","4","25"]},
 {"q":"Compute 19*21.","correct":"399","distractors":["389","409","419"]},
 {"q":"Compute 144 / 6 + 17*3.","correct":"75","distractors":["73","77","71"]},
 {"q":"Compute 2^8 + 2^6.","correct":"320","distractors":["288","256","384"]},
 {"q":"Compute 6! / 3!.","correct":"120","distractors":["720","20","6"]},
 {"q":"Sum of the integers 1 through 100.","correct":"5050","distractors":["5000","5100","4950"]},
 {"q":"Compute 1000 - 37*17.","correct":"371","distractors":["381","361","391"]},
 {"q":"Compute 45*11.","correct":"495","distractors":["485","505","455"]},
 {"q":"Compute 13^2 + 14^2.","correct":"365","distractors":["355","375","345"]},
 {"q":"How many seconds are in 2.5 hours?","correct":"9000","distractors":["8000","9500","7200"]},
 {"q":"Compute 7*8*9 / 6.","correct":"84","distractors":["72","96","63"]},
 {"q":"What is the 7th Fibonacci number (1,1,2,3,5,...)?","correct":"13","distractors":["11","8","21"]},
 {"q":"Compute 256 / 16 + 16*4.","correct":"80","distractors":["72","84","68"]},
 {"q":"Compute 3^4 - 2^5.","correct":"49","distractors":["51","47","59"]},
 {"q":"Compute gcd(60, 96).","correct":"12","distractors":["6","24","18"]},
 {"q":"Compute 20% of 350 plus 30% of 200.","correct":"130","distractors":["120","140","110"]},
 {"q":"Compute 11*12*13 / 22.","correct":"78","distractors":["66","84","72"]},
 {"q":"How many edges does a cube have?","correct":"12","distractors":["8","6","16"]},
 {"q":"Compute 5! + 4! + 3!.","correct":"150","distractors":["144","156","120"]},
 {"q":"Compute 1/2 + 1/3 + 1/6.","correct":"1","distractors":["5/6","7/6","2/3"]},
 {"q":"Compute sqrt(81) + cbrt(64).","correct":"13","distractors":["11","12","17"]},
 {"q":"In Python, len('hello world'.split()) gives?","correct":"2","distractors":["1","11","3"]},
 {"q":"In Python, sum([i*i for i in range(4)]) gives?","correct":"14","distractors":["9","30","16"]},
 {"q":"In Python, 'racecar'[::-1] == 'racecar' gives?","correct":"True","distractors":["False","error","None"]},
 {"q":"How many diagonals does a hexagon have?","correct":"9","distractors":["6","12","18"]},
 {"q":"Compute 17 + 17*0 + 17.","correct":"34","distractors":["0","17","51"]},
 {"q":"Compute 100 / 4 / 5.","correct":"5","distractors":["20","125","1"]},
 {"q":"A car goes 60 mph for 2h then 30 mph for 1h. Average speed?","correct":"50 mph","distractors":["45 mph","40 mph","55 mph"]},
 {"q":"Compute 2*3 + 4*5 - 6*7.","correct":"-16","distractors":["16","-12","-8"]},
]

LETTERS=["A","B","C","D"]
def build_choices(p, seed):
    rng=_r.Random(seed); opts=[p["correct"]]+list(p["distractors"]); rng.shuffle(opts)
    return opts, opts.index(p["correct"])
log("loading", MODEL_ID); t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
DEV=model.device
def _res(paths):
    for p in paths:
        cur=model; ok=True
        for q in p.split("."):
            if not hasattr(cur,q): ok=False;break
            cur=getattr(cur,q)
        if ok: return cur
LSTACK=_res(("model.layers","model.model.layers","model.language_model.layers")); NL=len(LSTACK)
NORM=_res(("model.norm","model.model.norm","model.language_model.norm")); LMH=model.get_output_embeddings()
def lyr(i): return LSTACK[i]
log(f"loaded {time.time()-t0:.0f}s NL={NL}")
SYS="You are a careful reasoning assistant. Think step by step, then answer."
def mcq(p,opts): return p["q"]+"\n"+"\n".join(f"({LETTERS[i]}) {c}" for i,c in enumerate(opts))
SUFFIX="\n\nThe answer is ("
def base_str(p,opts,think):
    msgs=[{"role":"system","content":SYS},{"role":"user","content":mcq(p,opts)}]
    try: return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=think)
    except TypeError: return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
def to_ids(s): return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
LET_TOK=[tok(l, add_special_tokens=False).input_ids[0] for l in LETTERS]
def llens(x):
    with torch.no_grad():
        return torch.log_softmax(LMH(NORM(x.reshape(1,-1).to(DEV, next(model.parameters()).dtype))).reshape(-1).float(), -1)
def cap_all(ids):
    res={}; hs=[lyr(L).register_forward_hook(lambda m,i,o,L=L: res.__setitem__(L,(o[0] if isinstance(o,tuple) else o)[0,-1,:].detach())) for L in range(NL)]
    with torch.no_grad(): model(ids,use_cache=False)
    for h in hs: h.remove(); return res
def argmax_letter(ids):
    with torch.no_grad(): lg=model(ids,use_cache=False).logits[0,-1].float()
    return int(np.argmax([float(lg[t]) for t in LET_TOK]))
FLIP_IDX=[7,10,13,14,16,19,23,25,31,33,34,37,46,49,58,59]
margin={L:[] for L in range(NL)}
for idx in FLIP_IDX:
    p=PROBLEMS[idx]; opts,ai=build_choices(p,idx)
    tp=base_str(p,opts,True); tids=to_ids(tp)
    with torch.no_grad():
        g=model.generate(input_ids=tids,max_new_tokens=THINK_MAX,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,attention_mask=torch.ones_like(tids))
    th=tok.decode(g[0,tids.shape[1]:],skip_special_tokens=False); th=th.split("</think>")[0] if "</think>" in th else th[:2000]
    cot_ids=to_ids(tp+th+"</think>"+SUFFIX); nocot_ids=to_ids(base_str(p,opts,False)+SUFFIX)
    ac=argmax_letter(cot_ids); an=argmax_letter(nocot_ids)
    res=cap_all(cot_ids)
    for L in range(NL):
        lp=llens(res[L]); margin[L].append(float(lp[LET_TOK[ac]]-lp[LET_TOK[an]]))
    log(f"idx{idx}: ans_cot={LETTERS[ac]} ans_nocot={LETTERS[an]} | margin L15 {np.mean([margin[15][-1]]):+.2f} L59 {margin[59][-1]:+.2f}")
prof={L: float(np.mean(margin[L])) for L in range(NL)}
SAE_L=[15,19,23,27,35,39,43,47,51,59,63]
log("=== logit-lens margin (ans_cot - ans_nocot) at the decision token, per layer ===")
for L in SAE_L: log(f"  L{L}: margin {prof[L]:+.3f}")
# where does the margin first exceed 0.5? (the consolidation onset)
onset=next((L for L in range(NL) if prof[L]>0.5), None)
log(f"margin onset (>0.5): L{onset} of {NL}")
print("OILAB_JSON_BEGIN"); print(json.dumps({"per_layer_margin":prof,"onset":onset,"NL":NL,"sae_layers":SAE_L})); print("OILAB_JSON_END", flush=True)
log("DONE")
