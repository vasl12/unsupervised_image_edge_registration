
import os
import nibabel as nib
import matplotlib.pyplot as plt

def visualise_md_dset(path, axis):
    dirs = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        dirs.extend(dirnames)
        break

    fig = plt.figure()
    title_pad = 10

    f = []
    # for every directory inside the parent directory
    for i, dir in enumerate(dirs):
        j = i%20
        dir_path = os.path.join(path, dir)
        filename = os.path.join(dir_path, dir+'.nii.gz')
        imgs = nib.load(filename)
        img = imgs.get_fdata()
        t1 = img[..., 1]
        t2 = img[..., 3]
        sizes = t1.shape
        z = int(sizes[axis] // 2)
        t1 = t1.take(z, axis=axis)  # (X, X)
        t2 = t2.take(z, axis=axis)  # (X, X)

        if (i%20==0 and i>=20) or i==len(dirs)-1:
            fig_save_path = os.path.join(path, f'_{i}.png')
            fig.savefig(fig_save_path, bbox_inches='tight', dpi=100)
            # plt.show()
            fig = plt.figure()

        ax = plt.subplot(4, 5, j+1)
        plt.imshow(t1, cmap='gray')
        plt.axis('off')
        plt.colorbar()
        # ax = plt.subplot(1, 2, i + 1)
        # plt.imshow(t2, cmap='gray')
        # plt.axis('off')
        # plt.colorbar()
        ax.set_title(dir, fontsize=10, pad=title_pad)

    print('a')








if __name__ == "__main__":
    data_path = '/home/pti/Documents/git/datasets/Task01_BrainTumour/imagesTs'
    visualise_md_dset(data_path, axis=2)