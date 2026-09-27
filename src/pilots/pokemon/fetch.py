import json, time, os, urllib.request, ssl, certifi
C = ssl.create_default_context(cafile=certifi.where())
def get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-pilot"}), context=C, timeout=30).read().decode()
ids, before = [], None
while len(ids) < 160:
    u = "https://replay.pokemonshowdown.com/search.json?format=gen9randombattle" + (f"&before={before}" if before else "")
    page = json.loads(get(u)); time.sleep(0.6)
    if not page: break
    ids += [r["id"] for r in page[:50] if (r.get("rating") or 0) >= 1600 and r["id"] not in ids]
    before = page[-1]["uploadtime"]
print(len(ids))
os.makedirs("data/logs", exist_ok=True)
for i in ids[:160]:
    p = f"data/logs/{i}.log"
    if os.path.exists(p): continue
    try:
        open(p, "w").write(get(f"https://replay.pokemonshowdown.com/{i}.log"))
    except Exception as e: print(i, e)
    time.sleep(0.6)
