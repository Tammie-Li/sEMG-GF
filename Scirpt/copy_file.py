

import os, shutil


def copy_files_dir_to_dir(src_dir_path, tar_dir_path):
    # 将一个路径文件夹下的所有文件全部拷贝到另一个文件夹下
    src_file_name = os.listdir(src_dir_path)
    
    for name in src_file_name:
        file_path = os.path.join(src_dir_path, name)
        shutil.copy(file_path, tar_dir_path)

if __name__ == "__main__":

    for subject_id in range(1, 11):
        for window_size_ms in [250, 500, 750]:
            tar_dir_path = os.path.join(os.getcwd(), "Result", "CrossSubjectExp", f"Sub{subject_id:>02d}", f"Window_size_{window_size_ms}ms")
            src_dir_path = os.path.join("D:\\CrossSubjectExp", f"Sub{subject_id:>02d}", f"Window_size_{window_size_ms}ms")
            copy_files_dir_to_dir(src_dir_path, tar_dir_path)