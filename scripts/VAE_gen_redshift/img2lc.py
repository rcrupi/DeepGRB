from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import seaborn as sns

with sns.plotting_context("talk"):
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    for i in [1, 2, 3]:
        g = Image.open('C:/Users/riccardo/Downloads/sample_GRB_images/fake_' + str(i) + '.JPG')
        g = g.resize((516, 128))
        # g.show()
        data = asarray(g)[:, 2:514, :]
        data_mono = data.sum(axis=2)
        data_lc = data_mono.sum(axis=0)
        data_lc = (data_lc - min(data_lc)) / (max(data_lc) - min(data_lc))
        ax[i-1].step(range(0, len(data_lc)), data_lc)
        ax[i-1].set_title('Fake lightcurve #' + str(i))
        if i == 3:
            ax[i-1].set_xlabel('Timestep')
        if i == 2:
            ax[i-1].set_ylabel('Normalized count rates')
    fig.tight_layout()

# plt.xlabel('Timestep')
# plt.ylabel('Normalized count rates')
# fig.suptitle('This is a somewhat long figure title', fontsize=16)
plt.show()

pass
