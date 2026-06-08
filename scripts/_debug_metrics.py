import json

with open('results/filter_benchmark/benchmark_results.json') as f:
    data = json.load(f)

for r in data['results']:
    m = r.get('metrics', {})
    agr = m.get('percentage_agreement')
    if agr is None or agr < 1.0:
        lbs = m.get('labels', [])
        cm = m.get('confusion_matrix', [])
        notes = r.get('inference_notes', {})
        print(f"=== {r['file']} ===")
        print(f"  agreement={agr}  n_agree={m.get('n_agree')}  n_agree_all={m.get('n_agree_all')}")
        print(f"  n_fix_gt={m.get('n_fix_in_gt')}  n_sac_gt={m.get('n_sac_in_gt')}")
        print(f"  labels: {lbs}")
        if cm and lbs:
            header = "              " + "".join(f"{l:>16}" for l in lbs)
            print(f"  Confusion matrix (rows=GT, cols=pred):")
            print("  " + header)
            for lbl, row in zip(lbs, cm):
                print(f"  {lbl:13}" + "".join(f"{v:>16}" for v in row))
        print()
