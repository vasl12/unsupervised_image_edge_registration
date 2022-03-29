import os
import shutil
import subprocess

# only used once to create the suitable data structure
def rearrage_dataset_whole(path):
    dirs = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        dirs.extend(dirnames)
        break

    f = []

    # for every directory inside the parent directory
    for dir in dirs:
        dir_path = os.path.join(path, dir)
        # find all files insider the directory
        for (dirpath, dirnames, filenames) in os.walk(dir_path):
            f.extend(filenames)
            break
            # for every file check if a dir exist and create it
        for file in f:
            # [0] because .nii.gz
            filename, file_extension = os.path.splitext(os.path.splitext(file)[0])
            file_dir = os.path.join(dir_path, filename)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
                shutil.move(os.path.join(dir_path, file), file_dir)

        f = []

# only used once to create the suitable data structure
def rearrage_dataset_one_dir(path, suffix):

    f = []

    dir_path = path
    # find all files insider the directory
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        f.extend(filenames)
        break
        # for every file check if a dir exist and create it
    for file in f:
        # [0] because .nii.gz
        filename, file_extension = os.path.splitext(os.path.splitext(file)[0])
        filename = filename.split(suffix)[0]
        file_dir = os.path.join(dir_path, filename)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
            shutil.move(os.path.join(dir_path, file), file_dir)
        f = []


# only used once to create the suitable data structure
def rearrage_dataset_move_up(path):
    dirs = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        dirs.extend(dirnames)
        break

    f = []

    # for every directory inside the parent directory
    for dir in dirs:
        dir_path = os.path.join(path, dir)
        # find all files insider the directory
        for (dirpath, dirnames, filenames) in os.walk(dir_path):
            f.extend(filenames)
            break
            # for every file check if a dir exist and create it
        for file in f:
            # [0] because .nii.gz
            filename, file_extension = os.path.splitext(os.path.splitext(file)[0])
            file_dir = os.path.join(dir_path, filename)
            if not os.path.exists(file_dir):
                # os.makedirs(file_dir)
                shutil.move(os.path.join(dir_path, file), os.path.join(path, file))
                # cmd_link = f'ls -s {os.path.join(dir_path, file)} {os.path.join(path, file)}'

                # subprocess.check_call(cmd_link, shell=True)
        f = []


# only used once to create the suitable data structure
def create_dataset_t1t2(path, t1t2_path, suffix):

    f = []
    dir_path = path

    # find all files insider the directory
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        f.extend(dirnames)
        break
        # for every file check if a dir exist and create it

    for file in f:
        # [0] because .nii.gz
        filename = f'{file}{suffix}.nii.gz'
        # filename, file_extension = os.path.splitext(os.path.splitext(file)[0])
        # filename = filename.split(suffix)[0]
        file_dir = os.path.join(t1t2_path, file)
        shutil.move(os.path.join(dir_path, file, filename), os.path.join(file_dir, filename))
    f = []


if __name__ == "__main__":
    data_path = '/home/pti/Documents/datasets/IXI/T2_Reg'
    t1t2_path = '/home/pti/Documents/datasets/IXI/T1_Reg'
    suffix = '-T2'
    create_dataset_t1t2(data_path, t1t2_path, suffix)
    # rearrage_dataset_one_dir(data_path, suffix)