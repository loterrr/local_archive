from __future__ import annotations
from datetime import datetime, timezone
import csv, json

def make_record(question, answer, contexts, params):
    retrieved=[]
    score_kind=params.get("score_kind", "retrieval")
    for c, score, dense, sparse in contexts:
        row={"chunk_id":c.chunk_id,"filename":c.filename,"page_number":c.page_number,"text":c.text,
             "score":float(score),"score_kind":score_kind,"dense_rank":dense,"sparse_rank":sparse}
        retrieved.append(row)
    return {"timestamp_utc":datetime.now(timezone.utc).isoformat(),"question":question,"answer":answer,
            "retrieved_segments":retrieved,"parameters":params,"llm_context":"\n\n".join(c.text for c,*_ in contexts)}

def to_json(records): return json.dumps(records,ensure_ascii=False,indent=2)

def to_csv(records):
    import io
    out=io.StringIO(); fields=["timestamp_utc","question","answer","retrieved_segments","parameters","llm_context"]
    w=csv.DictWriter(out,fieldnames=fields); w.writeheader()
    for r in records:
        row=dict(r); row["retrieved_segments"]=json.dumps(row["retrieved_segments"],ensure_ascii=False); row["parameters"]=json.dumps(row["parameters"],ensure_ascii=False); w.writerow(row)
    return out.getvalue()
