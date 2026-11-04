"""Convert one cohort of the NMI 3-scenario study (paper pipeline, committed gen_tacro.py via scripts/scen_rerun/gen_seeded.py;
virtual_cohort_{train,test}.csv: one visit per patient = the 7th dose from an empty body, 12 noisy samples at 0-24 h, true
AUC of the dosing interval) to the EB-VAE input format (PharmaNODE 2026-10-07).
SHORT timeline, t = 0 at the visit dose: one dose (the visit dose) at 0 and the visit's 12 observations. The six earlier doses
are NOT simulated: the drug on board at t = 0 is left to the EB-VAE's patient-specific initial condition (k_IC -> z0), as
in Baaz's own simulated design (central amount starting at v). This gives the EB-VAE the same information as the plain latent
ODE it is compared with (3 samples at 0, 1, 3 h + dose + formulation + CYP; no dosing history).
Columns: ID TIME DV AMT EVID ENC(1 = sample at 0, 1 or 3 h: the encoder's input) VISIT(=1) AUC(true AUC of the interval)
COV_IND(=0) TREATMENT(1 Advagraf, 2 Prograf) FORM CYP HTN(=HT/35; constant 1 in scenarios 1 and 3).
    convert_scen.py <virtual_cohort_*.csv> <out.csv>"""
import sys, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src)
rows = []
for i, g in d.groupby('ID', sort=True):
    p = g.iloc[0]; form = p.DRUG
    cov = dict(COV_IND=0, TREATMENT=2 if form == 'Prograf' else 1, FORM=float(form == 'Prograf'), CYP=float(p.CYP), HTN=float(p.HT) / 35.0)
    dose = g[(g.mdv == 1) & (g.TIME.astype(float) == 0.0)]
    assert len(dose) == 1, (i, len(dose))
    rows.append(dict(ID=int(i), TIME=0.0, DV=np.nan, AMT=float(dose.AMT.iloc[0]), EVID=1, ENC=0, VISIT=0, AUC=np.nan, **cov))
    for _, r in g[g.mdv == 0].sort_values('TIME').iterrows():
        t = float(r.TIME)
        rows.append(dict(ID=int(i), TIME=t, DV=float(r.DV), AMT=0.0, EVID=0, ENC=int(any(abs(t - e) < 1e-6 for e in (0.0, 1.0, 3.0))),
                         VISIT=1, AUC=float(r.AUC), **cov))
df = pd.DataFrame(rows).sort_values(['ID', 'TIME', 'EVID'], ascending=[True, True, False])
df.to_csv(out, sep=';', index=False)
obs = df[df.EVID == 0]
print(f'{out}: {df.ID.nunique()} patients, {len(obs)} observations ({obs.groupby("ID").size().unique().tolist()} per patient), '
      f'encoder points per patient {obs.groupby("ID").ENC.sum().unique().tolist()}, Prograf {int(df.groupby("ID").FORM.first().sum())}, '
      f'max time {df.TIME.max():.0f} h')
