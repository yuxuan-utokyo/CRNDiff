# R21: export the logged history of an offline wandb run (.wandb file) to CSV, read-only. Uses the installed wandb's own reader.
import csv, json, sys
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
src, dst = sys.argv[1], sys.argv[2]
ds = DataStore(); ds.open_for_scan(src); rows, kinds = [], {}
while True:
    data = ds.scan_data()
    if data is None: break
    r = pb.Record(); r.ParseFromString(data); k = r.WhichOneof("record_type"); kinds[k] = kinds.get(k, 0) + 1
    if k == "history":
        d = {}
        for it in r.history.item:
            key = it.key or "/".join(it.nested_key)
            try: d[key] = json.loads(it.value_json)
            except Exception: d[key] = it.value_json
        rows.append(d)
keys = sorted({k for d in rows for k in d})
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for d in rows: w.writerow({k: (v if not isinstance(v, (dict, list)) else json.dumps(v)) for k, v in d.items()})
print(json.dumps({"records": kinds, "history_rows": len(rows), "columns": keys}))
