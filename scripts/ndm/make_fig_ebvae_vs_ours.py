#!/usr/bin/env python3
"""EB-VAE versus the dose-event model on the regimen benchmark of Baaz et al. (linear simulator, sparse sampling), for the simplified
EB-VAE subsection of ndm.tex ( + "how they construct [the model] is the reason ... that is the thing I want to highlight" -> the
measured mechanism, 16/10). Same files and definitions as scripts/regimen/baaz_arm_split.py (copied, so nothing else is rewritten):
  a  architecture, in the notation of Baaz et al.: the three places where the arm difference can be taken for the effect of the dose
     in the EB-VAE as implemented (dose-conditioned prior learned as a population model; encoder reading the concentrations
     relative to the population prediction for the patient's own arm; dose as a learned input of the vector field; project notes
     0.00zz81) and how the dose-event model closes each; the two
     measurements (EB-VAE prior without the dose: unchanged, 0.76 vs 0.75, tacrolimus design, our model with an
     encoder that also reads the dose: 0.49 vs EB-VAE 0.47 under titration)
  b  who receives 800 mg in training: true elimination rate by arm, randomised vs titrated (seed-1 training draw, as
     results/regimen_dh/figures/test2_titration.png panel A)
  c  per training set (5 randomised + 3 titrated sparse seeds): the model's within-patient ratio AUC(850 mg)/AUC(350 mg), median over
     the 400 test patients, against the true exposure ratio between the two training arms (geometric mean AUC, 800-mg / 400-mg arm).
     Horizontal line: the true within-patient ratio; diagonal: a model that takes the arm ratio for the effect of the dose.
  d  consequence for a test patient observed on 800 mg and predicted at 350 mg: median signed AUC error, per seed.
Revision 17/10 (; backup .bak_newseeds): panels c and d also show the confirmation runs of
main-text Table 3 (new training sets: randomised seeds 5-13, seed 14 left out with its pair because the EB-VAE diverged; titrated
seeds 5-14). 18/10 (; backup .bak_openmarkers): all training sets
drawn alike. Bars in d: means over all training sets.
Writes figures/ebvae_vs_ours.{pdf,png} in the manuscript folder."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.getcwd(); R = f'{ROOT}/results/regimen_dh'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
sys.path.insert(0, EB)
import gen_regimens as G
OUT = os.environ.get('FIG_OUT') or f'{REPO}/figures/ebvae_vs_ours'   # 26/10: FIG_OUT=... to render elsewhere
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'mathtext.fontset': 'dejavusans', 'axes.linewidth': 0.7,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
INK, MUTED, GREY = '#222222', '#666666', '#8a8a8a'
EBC, OURS, HI = '#1b9e77', '#6a3d9a', '#d9822b'          # EB-VAE green (Fig. of the regimen tests), ours purple (step 3), 800 mg orange
SEEDS = {'ctrl': list(range(5)) + list(range(5, 14)), 'titr': list(range(3)) + list(range(5, 15))}   # 5-14: confirmation runs
NEW = lambda s: s >= 5


def path(model, cond, s):
    tag = f'N100_sparse_s{s}.json'
    if NEW(s):   # confirmation runs (queue_weekend.sh)
        sub = 'titr0.8' if cond == 'titr' else 'ctrl'
        return f'{EB}/res_titr/confirm/{sub}/{tag}' if model == 'ebvae' else f'{R}/confirm/{sub}/Bfixkp20n/{tag}'
    if model == 'ebvae':
        p = f'{EB}/res_titr/{"titr0.8" if cond == "titr" else "ctrl"}/{tag}'
        return p if os.path.exists(p) or cond == 'titr' else f'{EB}/res/N100_sparse' + (f'_s{s}' if s else '') + '.json'
    return f'{R}/{"titr0.8/" if cond == "titr" else ""}Bfixkp20n/{tag}'


def per_seed(model, cond, s):
    r = json.load(open(path(model, cond, s)))['individual']
    t = json.load(open(path('ebvae', cond, s)))['individual']        # true AUCs, as baaz_arm_split.truth
    tru = {g: np.array(t[g]['auc_pred_per_patient']) / (1 + np.array(t[g]['auc_err_per_patient'])) for g in ('amt350', 'amt850')}
    e = {g: np.array(r[g]['auc_err_per_patient']) for g in ('amt350', 'amt850')}
    q = (1 + e['amt850']) / (1 + e['amt350'])
    return dict(ratio=float(np.median(q * tru['amt850'] / tru['amt350'])), rel=float(np.median(q)),
                true=float(np.median(tru['amt850'] / tru['amt350'])), cross=100 * float(np.median(e['amt350'][1::2])))


def arm_ratio(cond, s):
    P = G.draw(200, np.random.default_rng(s))
    arm = G.titrate(P, 0.8, s)[0] if cond == 'titr' else np.where(np.arange(200) % 2 == 0, 400.0, 800.0)
    t, c = G.simulate(P, [0, 8, 16, 24], np.repeat(arm[:, None], 4, 1)); a = np.trapezoid(c, t, axis=1)
    return float(np.exp(np.mean(np.log(a[arm == 800])) - np.mean(np.log(a[arm == 400]))))


D = {(m, c, s): per_seed(m, c, s) for m in ('ebvae', 'ours') for c in SEEDS for s in SEEDS[c]}
AR = {(c, s): arm_ratio(c, s) for c in SEEDS for s in SEEDS[c]}
true_ratio = np.mean([D[('ebvae', c, s)]['true'] for c in SEEDS for s in SEEDS[c]])
lines = []
for m in ('ebvae', 'ours'):
    xs = np.log([AR[k] for k in AR]); ys = np.log([D[(m,) + k]['rel'] for k in AR])
    lines.append(f'{m}: slope of log(model/true) on log(arm ratio) {np.polyfit(xs, ys, 1)[0]:.2f}; model/true randomised '
                 f'{np.mean([D[(m, "ctrl", s)]["rel"] for s in SEEDS["ctrl"]]):.2f}, titrated {np.mean([D[(m, "titr", s)]["rel"] for s in SEEDS["titr"]]):.2f}; '
                 f'800->350 randomised {np.mean([D[(m, "ctrl", s)]["cross"] for s in SEEDS["ctrl"]]):+.1f}, titrated '
                 f'{np.mean([D[(m, "titr", s)]["cross"] for s in SEEDS["titr"]]):+.1f}')
for m in ('ebvae', 'ours'):
    for lab, keep in (('development', lambda s: not NEW(s)), ('new', NEW)):
        lines.append(f'  {m} {lab}: model/true randomised {np.mean([D[(m, "ctrl", s)]["rel"] for s in SEEDS["ctrl"] if keep(s)]):.2f}, titrated '
                     f'{np.mean([D[(m, "titr", s)]["rel"] for s in SEEDS["titr"] if keep(s)]):.2f}; 800->350 titrated '
                     f'{np.mean([D[(m, "titr", s)]["cross"] for s in SEEDS["titr"] if keep(s)]):+.1f}')
lines.append('arm ratios: ' + ' '.join(f'{c}{s} {AR[(c, s)]:.2f}' for (c, s) in AR) + f'; true within-patient ratio {true_ratio:.2f}')
print('\n'.join(lines))

fig = plt.figure(figsize=(6.4, 4.4801))   # 26/10: grey notes under panel a removed, the 5-unit band they used cropped (was 4.75 in)
gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.15, 1.0], wspace=0.48, left=0.07, right=0.985, top=0.45060, bottom=0.10602)   # same positions in inches as before

# ---------------------------------------------------------------- a: architecture, where the training arm can enter
from matplotlib.patches import FancyBboxPatch
A = fig.add_axes([0.0, 0.53012, 1.0, 0.46988]); A.set_xlim(0, 100); A.set_ylim(5.0, 44); A.axis('off')
A.text(1.0, 43.2, 'a   two architectures: where the arm difference can be attributed to the dose', fontsize=7.5,
       fontweight='bold', va='top')


def abox(x, y, w, h, title, body, ec, fc='#ffffff'):
    A.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', ec=ec, fc=fc, lw=0.9, zorder=2))
    A.text(x + w / 2, y + h - 1.2, title, ha='center', va='top', fontsize=6.4, fontweight='bold', color=INK, zorder=3)
    A.text(x + w / 2, y + h - 4.7, body, ha='center', va='top', fontsize=5.9, color=INK, zorder=3, linespacing=1.3)


def badge(x, y, k, ec, fc):
    A.text(x, y, str(k), ha='center', va='center', fontsize=6.2, fontweight='bold', color='white' if fc != '#ffffff' else ec, zorder=5,
           bbox=dict(boxstyle='circle,pad=0.18', fc=fc, ec=ec, lw=0.9))


def harrow(p0, p1, col):
    A.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='-|>', color=col, lw=0.9, shrinkA=0, shrinkB=0), zorder=4)


# Notation of Baaz et al. (their Fig. 1) and what their code does on this design: prior mean = network of the
# covariates and the first dose (x constant here -> one typical patient per arm), learned first as a population model (stage 1);
# encoder input = concentrations / stage-1 prediction under the patient's own regimen; dose = input u(t) of the vector field.
SIDES = (
    (1.0, 'EB-VAE (Baaz et al.)', 'latent: parameters $k=(k_D,k_{I_0})$, ODE state $z$', EBC, '#f2faf7', True,
     (r'Encoder $q_\phi(k\mid\tilde y,x)$', '$\\tilde y$: concentrations ÷ the\npopulation prediction for the\npatient\'s own arm'),
     (r'Prior $p_\psi(k\mid x,d)$', 'learned: one typical\npatient per training dose\n(population model, stage 1)'),
     (r'Neural ODE  $\dot z = f_\theta(z,k_D,u(t))$', 'the dose $u(t)$ is a learned input of the field,\nseen only at the two training doses')),
    (51.0, 'dose-event model (ours)', 'latent: physiology $p$, drug state $u$', OURS, '#f6f1fb', False,
     (r'Encoder $q(z_0\mid y/d,\,s)$', 'concentrations ÷ dose:\nthe arm is not visible\nfor a linear drug'),
     (r'Prior $N(0,I)$', 'fixed: one population\nfor every patient\nand every dose'),
     (r'Dose event  $u \leftarrow u + a\,b(p)$,  $\dot p = 0$', 'the dose is added to the drug state;\nthe flow never receives it')),
)
for x0, name, latent, col, fc, open_, enc, pri, dyn in SIDES:
    A.add_patch(FancyBboxPatch((x0, 6.0), 48, 32.5, boxstyle='round,pad=0,rounding_size=1.6', ec=col, fc=fc, lw=0.9, alpha=0.7, zorder=0))
    A.text(x0 + 47, 37.3, name, ha='right', va='top', fontsize=6.8, fontweight='bold', color=col)
    A.text(x0 + 2, 37.3, latent, ha='left', va='top', fontsize=5.8, color=MUTED)
    abox(x0 + 2, 19.5, 21, 13.0, *enc, col)
    abox(x0 + 25, 19.5, 21, 13.0, *pri, col)
    abox(x0 + 2, 7.5, 44, 9.5, *dyn, col)
    harrow((x0 + 12.5, 19.5), (x0 + 12.5, 17.0), INK); harrow((x0 + 35.5, 19.5), (x0 + 35.5, 17.0), INK)
    A.text(x0 + 24, 18.25, '$k$' if open_ else '$z_0$', fontsize=6.4, ha='center', va='center', color=INK)
    for k, (bx, by) in ((2, (x0 + 2.6, 31.9)), (1, (x0 + 25.6, 31.9)), (3, (x0 + 2.6, 16.4))):   # 1 prior, 2 encoder, 3 dynamics
        badge(bx, by, k, HI, HI) if open_ else badge(bx, by, k, col, '#ffffff')
# 26/10: the two grey notes under panel a ('filled: open. Tested: ...', 'outlined: closed. Tested: ...') removed

# ---------------------------------------------------------------- a: who gets 800 mg
ax = fig.add_subplot(gs[0])
P = G.draw(200, np.random.default_rng(1)); ke = P[:, 1]
arms = {'randomised': np.where(np.arange(200) % 2 == 0, 400.0, 800.0), 'titrated': G.titrate(P, 0.8, 1)[0]}
rng = np.random.default_rng(1)
for k, (cond, arm) in enumerate(arms.items()):
    for a, col, dx in ((400, GREY, -0.19), (800, HI, 0.19)):
        v = ke[arm == a]; xs = k + dx + rng.uniform(-0.08, 0.08, len(v))
        ax.scatter(xs, v, s=3, color=col, alpha=0.6, lw=0)
        ax.plot([k + dx - 0.13, k + dx + 0.13], [np.median(v)] * 2, color=INK, lw=1.3)
ax.set_yscale('log'); ax.set_xticks([0, 1], ['randomised', 'titrated']); ax.set_xlim(-0.5, 1.5)
ax.set_yticks([0.2, 0.5, 1, 2], ['0.2', '0.5', '1', '2']); ax.minorticks_off()
ax.set_ylabel('true elimination rate (1/h)', labelpad=1)
ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=3, color=GREY, label='400 mg arm'),
                   Line2D([], [], ls='', marker='o', ms=3, color=HI, label='800 mg arm')], fontsize=5.8, frameon=False,
          loc='upper left', bbox_to_anchor=(-0.02, 1.02), handletextpad=0.2, borderaxespad=0.2)
ax.set_title('b   who receives 800 mg', fontsize=7.5, loc='left', fontweight='bold')

# ---------------------------------------------------------------- b: dose response vs arm ratio
ax = fig.add_subplot(gs[1])
xx = np.linspace(0.95, 2.3, 10)
ax.plot(xx, xx, color=MUTED, lw=0.7, ls=(0, (3, 2)))
ax.text(2.22, 1.70, 'arm ratio attributed\nto the dose', fontsize=5.8, color=MUTED, ha='right', va='top')
ax.axhline(true_ratio, color=INK, lw=0.7, ls=':')
ax.text(0.97, true_ratio - 0.05, f'true ratio {true_ratio:.2f}', fontsize=5.8, color=INK, va='top')
for m, col in (('ebvae', EBC), ('ours', OURS)):
    for c, mk in (('ctrl', 'o'), ('titr', '^')):
        ax.scatter([AR[(c, s)] for s in SEEDS[c]], [D[(m, c, s)]['ratio'] for s in SEEDS[c]], s=16, marker=mk, color=col,
                   edgecolors='white', linewidths=0.4, zorder=3)
ax.set_xlabel('exposure ratio between the training arms', labelpad=1)
ax.set_ylabel('AUC(850 mg) / AUC(350 mg),\nsame patient', labelpad=1)
ax.set_xlim(0.95, 2.25); ax.set_ylim(0.85, 3.05)
ax.legend(handles=[Line2D([], [], ls='', marker='s', ms=4, color=EBC, label='EB-VAE'),
                   Line2D([], [], ls='', marker='s', ms=4, color=OURS, label='dose-event model'),
                   Line2D([], [], ls='', marker='o', ms=3.5, color=GREY, label='randomised arms'),
                   Line2D([], [], ls='', marker='^', ms=3.5, color=GREY, label='titrated arms'),
                   ],
          fontsize=5.6, frameon=False, loc='upper left', handletextpad=0.2, labelspacing=0.35, borderaxespad=0.1)
ax.set_title('c   learned dose response', fontsize=7.5, loc='left', fontweight='bold')

# ---------------------------------------------------------------- c: consequence
ax = fig.add_subplot(gs[2])
for j, (m, col) in enumerate((('ebvae', EBC), ('ours', OURS))):
    for k, c in enumerate(('ctrl', 'titr')):
        v = [D[(m, c, s)]['cross'] for s in SEEDS[c]]
        x0 = k + (j - 0.5) * 0.36
        ax.scatter(x0 + np.linspace(-0.09, 0.09, len(v)), v, s=12, color=col, edgecolors='white', linewidths=0.4, zorder=3)
        ax.plot([x0 - 0.12, x0 + 0.12], [np.mean(v)] * 2, color=col, lw=1.3)
ax.axhline(0, color=INK, lw=0.6, ls=':')
ax.set_xticks([0, 1], ['randomised', 'titrated']); ax.set_xlim(-0.5, 1.5)
ax.set_ylabel('error at 350 mg, patient\nobserved on 800 mg (%)', labelpad=1)
ax.set_title('d   predicting another dose', fontsize=7.5, loc='left', fontweight='bold')
for a_ in fig.axes[1:]:
    for s_ in ('top', 'right'):
        a_.spines[s_].set_visible(False)

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', dpi=300, facecolor='white')
open('results/ndm/fig_ebvae_vs_ours.txt', 'w').write('\n'.join(lines) + '\n')
print(f'wrote {OUT}.{{pdf,png}}')
