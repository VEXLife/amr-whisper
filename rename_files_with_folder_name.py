import os

def rename_files_with_folder_name(base_path):
    """
    遍历 base_path 下的所有子文件夹，将文件夹名添加到文件名的前面。

    参数:
        base_path (str): 根目录路径
    """
    # 遍历根目录下的所有子文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)

        # 仅处理文件夹
        if os.path.isdir(folder_path):
            # 遍历文件夹中的所有文件
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # 跳过非文件
                if not os.path.isfile(file_path):
                    continue
                
                # 构造新的文件名（文件夹名_原文件名）
                new_file_name = f"{folder_name}_{file_name}"
                new_file_path = os.path.join(folder_path, new_file_name)

                # 重命名文件
                os.rename(file_path, new_file_path)
                print(f"Renamed: {file_path} -> {new_file_path}")

# 使用示例
if __name__ == "__main__":
    base_path = "./test_data"  # 修改为你的 test_data 路径
    rename_files_with_folder_name(base_path)