import os
import nibabel as nib
import matplotlib.pyplot as plt


def vis_arb_dset(filename, axis):

    fig = plt.figure()
    title_pad = 10

    imgs = nib.load(filename)
    img = imgs.get_fdata()
    sizes = img.shape
    z = int(sizes[axis] // 2)
    z = 150
    img = img.take(z, axis=axis)  # (X, X)

    ax = plt.subplot(1, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    ax.set_title('IXI_T1', fontsize=10, pad=title_pad)
    plt.show()

    print('End.')



if __name__ == "__main__":
    data_path = '/home/pti/Documents/git/datasets/IXI Dataset/IXI-T1/IXI002-Guys-0828-T1.nii.gz'
    vis_arb_dset(data_path, axis=1)