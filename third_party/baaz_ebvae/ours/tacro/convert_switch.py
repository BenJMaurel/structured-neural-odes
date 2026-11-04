"""Convert a PharmaNODE switch-design cohort (gen_tacro_confound.py --switch-delay) to the EB-VAE input format: ONE
continuous timeline per patient (t = 0 at the visit-1 dose): d1 at 0; d2 from t = 24 h every tau up to the visit-2
dose at 24 + 24*delay (no extra dose inside an observed interval, as in the generator); observations = visit-1 samples
at REL and visit-2 samples at t2 + REL (the cohort's own noisy DV). Columns: ID TIME DV AMT EVID COV_IND(=0, unused)
TREATMENT(1 Advagraf, 2 Prograf) FORM CYP HTN(=HT/35) ENC (1 = visit-1 sample at 0, 1 or 3 h: the encoder's input)
plus truth columns VISIT, AUC (true AUC of that visit's interval) for scoring.
    convert_switch.py <cohort_dir> <out.csv> <split: train|test> [id_list_file]"""
import sys, json, numpy as np, pandas as pd
cdir, out, split = sys.argv[1], sys.argv[2], sys.argv[3]
T = pd.read_csv(f'{cdir}/confound_truth.csv').set_index('ID')
O = pd.read_csv(f'{cdir}/virtual_cohort_film_{split}.csv')
delay = float(T.switch_delay.iloc[0]); t2 = 24.0 + 24.0 * delay
ids = sorted(O.ID.unique())
if len(sys.argv) > 4: ids = [i for i in ids if i in set(json.load(open(sys.argv[4])))]
rows = []
for i in ids:
    tr = T.loc[i]; form = tr.DRUG; tau = 12.0 if form == 'Prograf' else 24.0
    cov = dict(COV_IND=0, TREATMENT=2 if form == 'Prograf' else 1, FORM=float(form == 'Prograf'),
               CYP=float(tr.CYP == 'expresser'), HTN=float(tr.HT) / 35.0)
    doses = [(0.0, float(tr.d1))] + [(t, float(tr.d2)) for t in np.arange(24.0, t2 + 1e-9, tau)]
    for t, a in doses:
        rows.append(dict(ID=int(i), TIME=float(t), DV=np.nan, AMT=a, EVID=1, ENC=0, VISIT=0, AUC=np.nan, **cov))
    g = O[(O.ID == i) & (O.mdv == 0)]
    for v, off in ((1, 0.0), (2, t2)):
        gv = g[g.VISIT == v].sort_values('TIME')
        for _, r in gv.iterrows():
            t = float(r.TIME); enc = int(v == 1 and any(abs(t - e) < 1e-6 for e in (0.0, 1.0, 3.0)))
            rows.append(dict(ID=int(i), TIME=off + t, DV=float(r.DV), AMT=0.0, EVID=0, ENC=enc, VISIT=v, AUC=float(r.AUC), **cov))
df = pd.DataFrame(rows).sort_values(['ID', 'TIME', 'EVID'], ascending=[True, True, False])
df.to_csv(out, sep=';', index=False)
obs = df[df.EVID == 0]
print(f'{out}: {df.ID.nunique()} patients, {len(obs)} observations ({obs.groupby("ID").size().unique().tolist()} per patient), '
      f'{int((df.EVID == 1).sum())} dose events ({df[df.EVID == 1].groupby("ID").size().unique().tolist()} per patient), '
      f'encoder points per patient {obs.groupby("ID").ENC.sum().unique().tolist()}, max time {df.TIME.max():.0f} h')
