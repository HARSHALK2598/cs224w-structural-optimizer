r"""The blog's three figures, written to report/figures/.

    python experiments/figures/make_figures.py

Figure 1 is an illustration. Figure 2 uses one driver's real driver-top3 test
dates. Figure 3 reads results/switch_history/switch_history.json (from
switch_history.py).
"""
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'report/figures'

# Reference palette (light mode):
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#b9b8b2'
GRID = '#e6e5e1'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'text.color': INK,
    'axes.labelcolor': INK2,
    'axes.edgecolor': MUTED,
    'xtick.color': INK2,
    'ytick.color': INK2,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
})


# ---------------------------------------------------------------- Figure 1
def box(ax, xy, text, color=BLUE, off=False, w=1.9, h=0.62, sub=None):
    x, y = xy
    face = '#f1f0ec' if off else '#ffffff'
    edge = MUTED if off else color
    ax.add_patch(
        FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.12',
                       linewidth=2, edgecolor=edge, facecolor=face, zorder=3))
    ax.text(x, y + (0.08 if sub else 0), text, ha='center', va='center',
            fontsize=11, color=MUTED if off else INK, zorder=4,
            fontweight='bold')
    if sub:
        ax.text(x, y - 0.17, sub, ha='center', va='center', fontsize=8.5,
                color=MUTED if off else INK2, zorder=4)


def link(ax, a, b, added=False, color=INK2, off=False, rad=0.0):
    ax.annotate(
        '', xy=b, xytext=a, zorder=2,
        arrowprops=dict(
            arrowstyle='-|>' if added else '-', lw=2,
            color=MUTED if off else color,
            linestyle=(0, (4, 3)) if added else '-', shrinkA=22, shrinkB=22,
            connectionstyle=f'arc3,rad={rad}'))


def figure1():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    pos = {
        'Driver': (0, 0),
        'Results': (-2.4, 1.8),
        'Standings': (0, 2.2),
        'Qualifying': (2.4, 1.8),
        'Races': (-4.7, 0.3),
        'Circuits': (-4.7, -1.6),
        'Constructors': (-2.4, -1.6),
    }
    schema = [('Driver', 'Results'), ('Driver', 'Standings'),
              ('Driver', 'Qualifying'), ('Results', 'Races'),
              ('Results', 'Constructors'), ('Races', 'Circuits')]

    ax = axes[0]
    for a, b in schema:
        link(ax, pos[a], pos[b])
    for name, xy in pos.items():
        box(ax, xy, name, color=ORANGE if name == 'Driver' else BLUE, w=2.3)
    ax.set_title('Before: the database graph\n(one edge per foreign key)',
                 fontsize=13, color=INK, pad=10)

    ax = axes[1]
    off = {'Constructors', 'Circuits'}
    for a, b in schema:
        link(ax, pos[a], pos[b], off=a in off or b in off)
    added = {'Similar driver': (4.5, 0.0), 'Earlier result': (-2.4, 3.4)}
    link(ax, added['Similar driver'], pos['Driver'], added=True, color=AQUA)
    ax.text(2.35, 0.45, 'similar drivers\n(KNN, behavioural)', ha='center',
            va='center', fontsize=9.5, color=INK2)
    link(ax, added['Earlier result'], pos['Results'], added=True,
         color=AQUA)
    ax.text(-4.2, 2.6, 'same driver,\nprevious race\n(temporal)', ha='center',
            va='center', fontsize=9.5, color=INK2)
    link(ax, pos['Races'], pos['Driver'], added=True, color=AQUA)
    ax.text(-3.45, -0.35, 'shortcut\n(two-hop)', ha='center', va='center',
            fontsize=9.5, color=INK2)
    for name, xy in {**pos, **added}.items():
        if name == 'Driver':
            box(ax, xy, name, color=ORANGE, sub='compressed (bottleneck)',
                w=2.7)
        elif name == 'Results':
            box(ax, xy, name, sub='some columns off', w=2.3)
        elif name in added:
            box(ax, xy, name, color=AQUA, w=2.3)
        else:
            box(ax, xy, name, off=name in off, w=2.3,
                sub='switched off' if name in off else None)
    ax.set_title('After: the structural optimizer\n(every table, column and '
                 'added edge type has an on/off switch)', fontsize=13,
                 color=INK, pad=10)

    for ax in axes:
        ax.set_xlim(-6.0, 5.8)
        ax.set_ylim(-2.2, 3.85)
        ax.set_aspect('equal')
        ax.axis('off')
    handles = [
        plt.Line2D([], [], color=INK2, lw=2, label='edge from a foreign key'),
        plt.Line2D([], [], color=AQUA, lw=2, ls=(0, (4, 3)),
                   label='added edge'),
        plt.Line2D([], [], color=MUTED, lw=2, label='switched off'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=11)
    fig.suptitle('What the four additions change around one driver '
                 '(illustration, rel-f1)', fontsize=12, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.08, top=0.84,
                        wspace=0.04)
    fig.savefig(OUT / 'fig1_method.png', dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
# Lewis Hamilton's 30 driver-top3 test dates (driverId 0):
TEST_DATES = [
    '2010-03-02', '2010-04-01', '2010-05-01', '2010-05-31', '2010-06-30',
    '2010-07-30', '2010-08-29', '2010-09-28', '2010-10-28', '2011-02-25',
    '2011-03-27', '2011-04-26', '2011-05-26', '2011-06-25', '2011-07-25',
    '2011-08-24', '2011-09-23', '2011-10-23', '2011-11-22', '2012-02-20',
    '2012-03-21', '2012-04-20', '2012-05-20', '2012-06-19', '2012-07-19',
    '2012-08-18', '2012-09-17', '2012-10-17', '2012-11-16', '2013-03-16'
]


def figure2():
    d = [datetime.fromisoformat(s) for s in TEST_DATES]
    cutoff = datetime(2010, 1, 1)
    start = datetime(2009, 1, 1)
    examples = [d[2], d[13], d[26]]  # 2010-05, 2011-06, 2012-09
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, title, full in [
        (axes[0], 'Full-history test (used by the paper\'s code)', True),
        (axes[1], 'RelBench cutoff test (RelBench standard)', False),
    ]:
        for row, t in enumerate(examples):
            y = len(examples) - row
            end = t if full else cutoff
            ax.barh(y, end - start, left=start, height=0.5, color=BLUE,
                    alpha=0.85, zorder=2)
            ax.barh(y, timedelta(days=30), left=t, height=0.5, color=ORANGE,
                    zorder=3)
            if not full and t > cutoff:
                ax.plot([cutoff, t], [y, y], color=MUTED, lw=1.5,
                        ls=(0, (2, 2)), zorder=1)
                gap = (t - cutoff).days / 365.25
                ax.text(t + timedelta(days=45), y,
                        f'{gap:.1f} years with no data', ha='left',
                        va='center', fontsize=9, color=INK2)
            ax.text(start - timedelta(days=12), y,
                    f'prediction on {t:%b %Y}', ha='right', va='center',
                    fontsize=10, color=INK)
        ax.axvline(cutoff, color=INK2, lw=1.2, zorder=4)
        ax.set_title(title, loc='left', fontsize=12, color=INK)
        ax.set_yticks([])
        ax.set_ylim(0.4, 3.8)
        for side in ['left', 'right', 'top']:
            ax.spines[side].set_visible(False)
        ax.grid(axis='x', color=GRID, zorder=0)
    axes[1].text(cutoff, 3.6, ' test period starts (1 Jan 2010)',
                 fontsize=9, color=INK2, va='center')
    # The 30 monthly test dates of this driver:
    axes[0].plot(d, [3.62] * len(d), '|', color=ORANGE, ms=8, mew=1.5)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[1].set_xlim(start, datetime(2013, 6, 1))
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.85,
                      label='rows the model may read'),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE,
                      label='the 30 days being predicted'),
        plt.Line2D([], [], color=ORANGE, marker='|', ls='', ms=8, mew=1.5,
                   label='this driver\'s 30 test predictions'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=10)
    fig.suptitle('Three of one driver\'s monthly test predictions '
                 '(Lewis Hamilton, driver-top3)', fontsize=13, x=0.5)
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.14, top=0.88,
                        hspace=0.35)
    fig.savefig(OUT / 'fig2_test_setups.png', dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def figure3():
    path = ROOT / 'experiments/results/switch_history/switch_history.json'
    history = json.loads(path.read_text())['history']
    epochs = [h['epoch'] for h in history]
    keys = [k for k in history[0] if k != 'epoch']
    kinds = [('table', 'table switch', ORANGE),
             ('column', 'column switch', BLUE),
             ('edge', 'added-edge-type switch', AQUA)]
    fig, ax = plt.subplots(figsize=(11, 6))
    for thr in [0.5, 0.7, 0.78, 0.82, 0.86, 0.9]:
        ax.axhline(thr, color=MUTED, lw=1, zorder=1)
        ax.text(50.6, thr, f'cut-off {thr:g}', va='center', fontsize=9,
                color=INK2)
    floor = 1 / (1 + math.exp(-(2.0 - 300 * 1e-3)))
    ax.axhline(floor, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(1, floor - 0.018, f'lowest value reachable in 300 training '
            f'steps ({floor:.3f})', fontsize=9, color=INK, va='top')
    for prefix, label, color in kinds:
        ks = [k for k in keys if k.startswith(prefix + '/')]
        for i, k in enumerate(ks):
            ax.plot(epochs, [h[k] for h in history], color=color, lw=1,
                    alpha=0.6, zorder=2,
                    label=f'{label} ({len(ks)})' if i == 0 else None)
    ax.text(25, 0.66, 'All switches start at 0.88 and never go below 0.846.\n'
            'A cut-off inside that narrow band switches some of them off;\n'
            'a cut-off below it (0.82 or lower) keeps every one on.',
            ha='center', va='center', fontsize=11, color=INK,
            linespacing=1.5)
    ax.set_xlim(0, 50)
    ax.set_ylim(0.45, 0.95)
    ax.set_xlabel('training epoch (6 optimizer steps each)')
    ax.set_ylabel('switch probability\n(the switch is on if above the '
                  'cut-off)')
    ax.set_title('The switches barely move during training, so the chosen '
                 'cut-off decides what is pruned', loc='left', fontsize=12)
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)
    leg = ax.legend(loc='lower left', frameon=False, fontsize=10,
                    title='each line is one switch')
    leg.get_title().set_color(INK2)
    for line in leg.get_lines():
        line.set_linewidth(2.5)
        line.set_alpha(1)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.11)
    fig.savefig(OUT / 'fig3_switches.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2()
    figure3()
