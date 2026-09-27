"""Write ds_{FMT}_slim.pkl with only the fields the text models need (keeps row order, so test indices match)."""
import os, pickle, sys
K = {"battle", "turn", "split", "y", "y_idx", "y_sw", "text", "htext", "cands", "cand_raw", "raw", "opp_cls", "cand_sw"}
d_ = sys.argv[1]; fmt_ = sys.argv[2] if len(sys.argv) > 2 else "gen9randombattle"
d = pickle.load(open(os.path.join(d_, f"ds_{fmt_}.pkl"), "rb"))
out = {t: [{k: v for k, v in r.items() if k in K} for r in rows] for t, rows in d.items()}
pickle.dump(out, open(os.path.join(d_, f"ds_{fmt_}_slim.pkl"), "wb"))
