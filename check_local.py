import matplotlib.pyplot as plt
import numpy as np
import torch

title = 'CIFAR100 \nLow Heterogeneity'
axisfont_minor = 12
axisfont_major = 20
label_font = 18
title_font = 24
b = 12
tag = 'output/runs/31_CIFAR100_label_resnet18_1_100_0.1_non-iid-50_dynamic_'
base = f'{tag}e1_bn_1_1_real_world.pt'
compare = f'{tag}a6-b10-c11-d18-e55_bn_1_1_real_world.pt'
acc = torch.load(compare)

x = np.array([i[0][0]['Local-Accuracy'] for i in acc])
x += np.random.normal(0, 1, x.shape)
x = np.clip(x, 0, 100)

fig, ax = plt.subplots()
ax.tick_params(axis='both', which='major', labelsize=axisfont_major)
ax.tick_params(axis='both', which='minor', labelsize=axisfont_minor)

n, bins, patches = plt.hist(x, bins=b, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.75)

n = n.astype('int')  # it MUST be integer
# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.get_cmap('Oranges')(n[i] / max(n)))

acc = torch.load(base)

x = np.array([i[0][0]['Local-Accuracy'] for i in acc[0]])
x += np.random.normal(0, 1, x.shape)
x = np.clip(x, 0, 100)

n, bins, patches = plt.hist(x, bins=b, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

n = n.astype('int')  # it MUST be integer
# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.get_cmap('Blues')(n[i] / max(n)))

plt.title(f'{title}', fontsize=title_font)
plt.xlabel('Accuracy', fontsize=label_font)
plt.ylabel('Counts', fontsize=label_font)

# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.size'] = label_font
# plt.rcParams['axes.linewidth'] = 1.0
# plt.show()

plt.savefig(f'{tag}a6-b10-c11-d18-e55_bn_1_1_real_world_accuracy_distribution.pdf', bbox_inches="tight")
# 'Accent', 'Accent_r', 'Blues', 'Blues_r',
# 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
