import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

df = pd.read_csv("C:/Users/gr3g2/Desktop/MUSIT_MDM/sailboat/output_detected_per_second_30.csv")
df['timestep'] = range(len(df))

COLOR_FULL = '#FF7F0E'   # vivid orange
COLOR_P3   = '#1F77B4'   # muted blue
MARKER_FULL = 'D'        # diamond
MARKER_P3   = 'o'        # circle

full = df[df['signal'] == 'FULL']
p3   = df[df['signal'] == 'P3']

# seaborn's default font and text proportions, scaled up for the 18 x 9 in figure
sns.set_theme(style='white', font_scale=2.2, rc={'axes.linewidth': 2.0, 'figure.dpi': 300})

fig, ax1 = plt.subplots(figsize=(18, 9))

fig.patch.set_facecolor('white')
ax1.set_facecolor('white')

ax2 = ax1.twinx()

ax1.scatter(p3['timestep'],   p3['distance_m'],   color=COLOR_P3,   marker=MARKER_P3,   s=120, alpha=0.85, zorder=3, linewidths=0)
ax1.scatter(full['timestep'], full['distance_m'], color=COLOR_FULL, marker=MARKER_FULL, s=150, alpha=0.95, zorder=4, linewidths=0)
ax1.axhline(30, color='#444444', linestyle='--', linewidth=2.0, alpha=0.6, zorder=2)
ax1.text(1, 31.2, 'dist = 30 m', color='#444444', va='bottom')
ax1.set_ylabel('Distance (m)', color='#222222')
ax1.set_xlabel('Timestep (s)', color='#222222')
ax1.tick_params(axis='y', labelcolor='#222222')
ax1.set_ylim(0, 45)

ax2.scatter(p3['timestep'],   p3['closure_rate_ms'],   color=COLOR_P3,   marker='v', s=100, alpha=0.60, zorder=3, linewidths=0)
ax2.scatter(full['timestep'], full['closure_rate_ms'], color=COLOR_FULL, marker='^', s=130, alpha=0.80, zorder=4, linewidths=0)
ax2.axhline(-0.5, color='#888888', linestyle=':', linewidth=2.0, alpha=0.7, zorder=2)
ax2.text(1, -0.62, 'rate = −0.5 m/s', color='#666666', va='top')
ax2.set_ylabel('Closure Rate (m/s)', color='#222222')
ax2.tick_params(axis='y', labelcolor='#222222')
ax2.set_ylim(-3.5, 2.5)

# FULL criterion: distance < 30 m AND closure rate < -0.5 m/s
full_ts = full['timestep'].values
if len(full_ts):
    groups, start, prev = [], full_ts[0], full_ts[0]
    for t in full_ts[1:]:
        if t - prev > 1:
            groups.append((start, prev))
            start = t
        prev = t
    groups.append((start, prev))
    for (s, e) in groups:
        ax1.axvspan(s - 0.5, e + 0.5, alpha=0.08, color=COLOR_FULL, zorder=1)

leg_handles = [
    mpatches.Patch(facecolor=COLOR_P3,   edgecolor='none', label='P3 signal'),
    mpatches.Patch(facecolor=COLOR_FULL, edgecolor='none', label='FULL signal'),
    mpatches.Patch(facecolor=COLOR_FULL, alpha=0.20, edgecolor='none', label='FULL criterion zone'),
]
fig.legend(handles=leg_handles, loc='lower center', bbox_to_anchor=(0.5, -0.12),
           ncol=4, framealpha=0.9, edgecolor='#cccccc',
           columnspacing=1.0, handlelength=1.4)

fig.suptitle(
    'Signal Classification over Time\n'
    'FULL criterion: distance < 30 m  AND  closure rate < −0.5 m/s',
    color='#111111', ha='center', y=1.02
)

ax1.grid(True, linestyle='--', linewidth=0.4, color='#cccccc', alpha=0.7)
ax1.set_axisbelow(True)

for spine in ax1.spines.values():
    spine.set_linewidth(0.8); spine.set_color('#555555')
for spine in ax2.spines.values():
    spine.set_linewidth(0.8); spine.set_color('#555555')

plt.tight_layout(pad=1.2)
plt.savefig('signal_classification.pdf',
            format='pdf', bbox_inches='tight',
            metadata={'Title': 'Signal Classification over Time',
                      'Author': 'Generated plot',
                      'Subject': 'FULL vs P3 signal — distance and closure rate'})
print("Saved: signal_classification.pdf")