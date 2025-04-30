import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Wayland issues

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# define groups from manual seeding
manual_groups = {
    'group1': ['jimbosmx', 'hintz', 'spencer', 'swagman', 'paranoiaboi', 'masongos', 'grady', 'jjk.', 'chezmix', 'inglomi', 'jellyslosh', 'eesa'],
    'group2': ['wdrm', 'senpi', 'janus5k', 'tayman', 'emcat', 'pilot'],
    'group3': ['mxl100', 'snowstorm', 'datcoreedoe', 'werdwerdus', 'cathadan', 'shinobee'],
    'group4': ['big matt', 'mesyr', 'zom585', 'ctemi', 'zephyrnobar', 'sydia'],
    'group5': ['kren', 'xosen', 'sweetjohnnycage', 'cheesecake', 'ditz', 'enderstevemc'],
    'group6': ['jiajiabinks', 'meeko', 'auby', 'arual', 'noir', 'dali'],
    'group7': ['momosapien', 'dogfetus', 'peter', 'jokko', 'butterbutt', 'jewel', 'beatao', 'maverick'],
}

# define a color per manual group
group_colors = {
    'group1': '#f4cccc',
    'group2': '#fce5cd',
    'group3': '#d9ead3',
    'group4': '#cfe2f3',
    'group5': '#c9daf8',
    'group6': '#d9d2e9',
    'group7': '#ead1dc',
}

# reverse lookup: player -> color (based on manual group)
player_colors = {player: group_colors[group] for group, members in manual_groups.items() for player in members}

# actual and algo groupings (same format)
actual_groups = {
    'group1': ['jimbosmx', 'hintz', 'spencer', 'swagman', 'paranoiaboi', 'masongos', 'grady', 'jjk.', 'chezmix', 'inglomi', 'jellyslosh', 'eesa', 'wdrm'],
    'group2': ['eesa', 'senpi', 'janus5k', 'tayman', 'emcat', 'pilot'],
    'group3': ['mxl100', 'snowstorm', 'datcoreedoe', 'werdwerdus', 'cathadan', 'big matt'],
    'group4': ['shinobee', 'zom585', 'ctemi', 'mesyr', 'zephyrnobar', 'sydia'],
    'group5': ['kren', 'xosen', 'sweetjohnnycage', 'cheesecake', 'ditz', 'enderstevemc'],
    'group6': ['jiajiabinks', 'meeko', 'momosapien', 'auby', 'arual', 'dogfetus'],
    'group7': ['noir', 'dali', 'peter', 'jokko', 'butterbutt', 'jewel', 'beatao', 'maverick'],
}

algo_groups = {
    'group1': ['paranoiaboi', 'chezmix', 'spencer', 'inglomi', 'hintz', 'jjk.', 'swagman', 'jimbosmx', 'tayman', 'masongos', 'janus5k', 'grady'],
    'group2': ['eesa', 'senpi', 'wdrm', 'pilot', 'emcat', 'jellyslosh'],
    'group3': ['snowstorm', 'datcoreedoe', 'werdwerdus', 'cathadan', 'mesyr', 'big matt'],
    'group4': ['shinobee', 'mxl100', 'ctemi', 'zom585', 'zephyrnobar', 'xosen'],
    'group5': ['sweetjohnnycage', 'sydia', 'ditz', 'cheesecake', 'kren', 'jiajiabinks'],
    'group6': ['meeko', 'enderstevemc', 'momosapien', 'auby', 'dogfetus', 'noir'],
    'group7': ['arual', 'dali', 'jokko', 'peter', 'butterbutt', 'jewel', 'maverick', 'beatao'],
}

# layout
group_labels = ['group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7']
sections = [('manual seeding', manual_groups), ('actual placement', actual_groups), ('algo iteration 5', algo_groups)]


fig, ax = plt.subplots(figsize=(16, 9))
ax.set_axis_off()

cell_width = 1.2
cell_height = 0.4

for row_idx, (section_title, group_data) in enumerate(sections):
    y_base = -row_idx * 10
    ax.text(-1.2, y_base + 0.5, section_title, fontsize=12, fontweight='bold')

    for col_idx, group in enumerate(group_labels):
        members = group_data.get(group, [])
        for i, name in enumerate(members):
            x = col_idx * cell_width
            y = y_base - i * cell_height
            color = player_colors.get(name, 'white')
            rect = Rectangle((x, y), cell_width, cell_height, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(x + cell_width / 2, y + cell_height / 2, name, ha='center', va='center', fontsize=8)

        ax.text(col_idx * cell_width + cell_width / 2, y_base + 0.5, group, ha='center', fontsize=10, weight='bold')

plt.title("Comparison of Manual Seeding, Actual Placement, and Algo Iteration 5", fontsize=14, weight='bold')

# Save to file instead of showing
plt.savefig("group_comparison.png", dpi=300, bbox_inches='tight')
