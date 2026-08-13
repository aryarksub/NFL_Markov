import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ABS_MAP_6 = {
    'Touchdown':         'TD',
    'Field goal':        'FG',
    'Punt':              'Punt',
    'End of half':       'EndHalf',
    'Turnover':          'Turnover',
    'Safety':            'Turnover',
    'Opp touchdown':     'Turnover',
    'Turnover on downs': 'Downs',
    'Missed field goal': 'Downs',
}

ABS_ORDER = ['TD', 'FG', 'Punt', 'Turnover', 'Downs', 'EndHalf']
ABS_COLORS = {
    'TD':       '#1D9E75',
    'FG':       '#E8A33D',
    'Punt':     '#378ADD',
    'Turnover': '#C0392B',
    'Downs':    '#8E44AD',
    'EndHalf':  '#95A5A6',
}

print("=== fixed_drive_result -> 6 absorbing states mapping check ===")
vc = df['fixed_drive_result'].value_counts(dropna=False)
print(vc.to_string())

unmapped = set(df['fixed_drive_result'].dropna().unique()) - set(ABS_MAP_6.keys())
print("\nunmapped drive results:", unmapped if unmapped else "(none)")

drive_level = (df.dropna(subset=['fixed_drive_result', 'game_id', 'fixed_drive'])
                 .groupby(['game_id', 'fixed_drive'])['fixed_drive_result']
                 .first()
                 .reset_index())
drive_level['abs6'] = drive_level['fixed_drive_result'].map(ABS_MAP_6)

abs_counts = drive_level['abs6'].value_counts().reindex(ABS_ORDER, fill_value=0)
total = abs_counts.sum()

print("\n=== 6 absorbing state distribution (drive-level) ===")
print(f"Total drives: {total:,}\n")
for s in ABS_ORDER:
    c = abs_counts[s]
    print(f"  {s:<10} {c:>8,}  ({100*c/total:5.1f}%)  {'#'*int(50*c/total)}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

bars = axes[0].bar(ABS_ORDER, abs_counts.values,
                   color=[ABS_COLORS[s] for s in ABS_ORDER])
axes[0].set_ylabel('drive count')
axes[0].set_title('Drive outcomes — 6 absorbing states (count)', fontweight='bold')
for b, s in zip(bars, ABS_ORDER):
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height(),
                 f'{abs_counts[s]:,}', ha='center', va='bottom', fontsize=9)

axes[1].pie(abs_counts.values, labels=ABS_ORDER, autopct='%1.1f%%',
            colors=[ABS_COLORS[s] for s in ABS_ORDER], startangle=90)
axes[1].set_title('Drive outcomes — 6 absorbing states (share)', fontweight='bold')

plt.tight_layout()
plt.savefig('absorbing6_distribution.png', dpi=130, bbox_inches='tight')
plt.show()
