import os
import re

def get_file_groups(folder):
    file_groups = {}
    for file_name in os.listdir(folder):
        match = re.match(r'^(\d+)\.\w+$', file_name)
        if match:
            file_number = match.group(1)
            if file_number in file_groups:
                file_groups[file_number].append(file_name)
            else:
                file_groups[file_number] = [file_name]
    return file_groups

def delete_incomplete_groups(folder, file_groups, required_count=7):
    for file_number, files in file_groups.items():
        if len(files) < required_count:
            print(f"Deleting files for group {file_number} as it has only {len(files)} files.")
            for file in files:
                os.remove(os.path.join(folder, file))

def main(folder):
    file_groups = get_file_groups(folder)
    delete_incomplete_groups(folder, file_groups)

if __name__ == "__main__":
    folder_path = "/home/mc/proj/PC2WF/simulateRoof_data/patches_50_noise_sigma0.01clip0.01/train"
    main(folder_path)
    files = os.listdir(folder_path)
    print(len(files))
