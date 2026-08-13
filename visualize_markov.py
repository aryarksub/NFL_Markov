"""
NFL Absorbing Markov Chain — 결과 시각화
전제: result (DataFrame), transient_states, state_idx, B, EP, expected_plays,
      ABS_STATES, N, Q, R 이 이미 계산돼 있음.

result 컬럼: 'state', 'EP', 'exp_plays', 그리고 각 ABS_* 흡수확률.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# ─────────────────────────────────────────────────────────────
# 헬퍼: state 튜플에서 yardline 구간의 대표 수치(중점) 뽑기
# yardline_100 기준: 값이 클수록 자기 진영(먼 곳), 작을수록 상대 골라인
# ─────────────────────────────────────────────────────────────
def yl_midpoint(label):
    if '-' in label:
        lo, hi = label.split('-')
        return (int(lo) + int(hi)) / 2
    return float(label)

# 1st & 10만 뽑아서 필드 위치별로 정렬 (Goldner Figure 1/3 재현용)
def get_first_and_ten(result):
    rows = []
    for _, r in result.iterrows():
        d, g, yl = r['state']
        if d == 1 and g == '10':
            rows.append({
                'yl_mid': yl_midpoint(yl),
                'EP': r['EP'],
                'exp_plays': r['exp_plays'],
                **{a: r[a] for a in ABS_STATES}
            })
    df = pd.DataFrame(rows).sort_values('yl_mid')
    return df

fat = get_first_and_ten(result)
# x축: 상대 골라인까지 거리 (yardline_100). 뒤집으면 "우리 진영→상대 진영"
x = fat['yl_mid'].values

# ─────────────────────────────────────────────────────────────
# Figure: 2x2 패널
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('NFL Drive Absorbing Markov Chain — Results (1st & 10)',
             fontsize=15, fontweight='bold')

# ── 패널 1: Expected Points 곡선 (Goldner Fig 3 재현) ──────────
ax = axes[0, 0]
ax.plot(x, fat['EP'].values, 'o-', color='#534AB7', markersize=5, linewidth=2)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xlabel('Yards from opponent goal line (yardline_100)')
ax.set_ylabel('Expected Points')
ax.set_title('Expected Points by field position', fontweight='bold')
ax.invert_xaxis()  # 상대 골라인(작은 값)을 오른쪽으로 → 전진 방향
ax.grid(alpha=0.3)

# ── 패널 2: 흡수 확률 stacked area (Goldner Fig 1 재현) ────────
ax = axes[0, 1]
# 주요 결과만 골라서 쌓기 (나머지는 묶음)
main_abs = ['ABS_TD', 'ABS_FG', 'ABS_PUNT', 'ABS_TO', 'ABS_DOWNS']
colors = ['#1D9E75', '#EF9F27', '#378ADD', '#E24B4A', '#D85A30']
other = 1 - fat[main_abs].sum(axis=1)
stack_data = [fat[a].values for a in main_abs] + [other.values]
stack_labels = ['Touchdown', 'Field goal', 'Punt', 'Turnover',
                'Downs', 'Other']
stack_colors = colors + ['#B4B2A9']
ax.stackplot(x, *stack_data, labels=stack_labels, colors=stack_colors,
             alpha=0.85)
ax.set_xlabel('Yards from opponent goal line')
ax.set_ylabel('Absorption probability')
ax.set_title('Drive outcome probabilities', fontweight='bold')
ax.invert_xaxis()
ax.set_ylim(0, 1)
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

# ── 패널 3: 기대 play 수 ──────────────────────────────────────
ax = axes[1, 0]
ax.plot(x, fat['exp_plays'].values, 's-', color='#0F6E56',
        markersize=5, linewidth=2)
ax.set_xlabel('Yards from opponent goal line')
ax.set_ylabel('Expected plays until drive ends')
ax.set_title('Expected number of plays remaining', fontweight='bold')
ax.invert_xaxis()
ax.grid(alpha=0.3)

# ── 패널 4: EP heatmap (down × field position) ────────────────
ax = axes[1, 1]
# down 1-4 × yardline 구간별 EP 격자 만들기
yl_labels_ordered = sorted(
    set(yl for (_, _, yl) in transient_states),
    key=yl_midpoint, reverse=True  # 자기 진영→상대 진영
)
grid = np.full((4, len(yl_labels_ordered)), np.nan)
for _, r in result.iterrows():
    d, g, yl = r['state']
    if g == '10':  # distance 고정 (10야드)해서 down×위치만 보기
        col = yl_labels_ordered.index(yl)
        grid[d - 1, col] = r['EP']
im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=7)
ax.set_yticks(range(4))
ax.set_yticklabels([f'{d}st/nd/rd/th' for d in [1, 2, 3, 4]])
# x축 라벨은 너무 많으니 일부만
step = max(1, len(yl_labels_ordered) // 10)
ax.set_xticks(range(0, len(yl_labels_ordered), step))
ax.set_xticklabels([yl_labels_ordered[i] for i in
                    range(0, len(yl_labels_ordered), step)],
                   rotation=45, ha='right', fontsize=7)
ax.set_xlabel('Field position (own → opponent)')
ax.set_ylabel('Down')
ax.set_title('EP heatmap (distance = 10)', fontweight='bold')
plt.colorbar(im, ax=ax, label='Expected Points')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/mnt/user-data/outputs/markov_results.png', dpi=130,
            bbox_inches='tight')
print("저장 완료: markov_results.png")
plt.show()
