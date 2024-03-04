#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/19 10:01
# @Author  : chongqing.zeng@sophgo.com
# @Project: PerfAI
import os
import shutil

# 获取当前目录
current_directory = os.getcwd()

# 遍历当前目录下的所有子文件夹
for folder_name in os.listdir(current_directory):
    folder_path = os.path.join(current_directory, folder_name)

    # 检查是否是文件夹
    if os.path.isdir(folder_path):
        # 创建一个字典用于存储相同NUMBER的文件路径
        number_files = {}

        # 遍历子文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # import pdb
            # pdb.set_trace()
            # 检查文件名是否符合规则 ins-NUMBER.*
            if file_name.startswith("ins-") and file_name.count('.') == 2:
                # import pdb
                # pdb.set_trace()
                _, number_part = file_name.split("-")
                number, engine, core_id = number_part.split(".")
                number = int(number)

                # 将文件移动到对应的文件夹中
                new_folder_path = os.path.join(current_directory, f"{folder_name}_{number}")
                os.makedirs(new_folder_path, exist_ok=True)
                new_name = f"{folder_name}_{number}" + f"-0.{engine}.{core_id}"
                # import pdb
                # pdb.set_trace()
                # 使用 os.rename() 方法重命名文件
                # os.rename(file_path, os.path.join(folder_path, new_name))
                shutil.move(file_path, os.path.join(new_folder_path, new_name))
            
            if file_name.startswith("ins-") and file_name.count('.') == 3:
                # import pdb
                # pdb.set_trace()
                _, number_part = file_name.split("-")
                number, engine, core_id, txt = number_part.split(".")
                number = int(number)
                if(txt == "txt"):
                    # 将文件移动到对应的文件夹中
                    new_folder_path = os.path.join(current_directory, f"{folder_name}_{number}")
                    os.makedirs(new_folder_path, exist_ok=True)
                    new_name = f"{folder_name}_{number}" + f"-0.{engine}.{core_id}.{txt}"
                    # import pdb
                    # pdb.set_trace()
                    # 使用 os.rename() 方法重命名文件
                    # os.rename(file_path, os.path.join(folder_path, new_name))
                    shutil.move(file_path, os.path.join(new_folder_path, new_name))

    try:
        shutil.rmtree(folder_path)
        print(f"文件夹 '{folder_path}' 已成功删除")
    except OSError as e:
        print(f"Error: {folder_path} - {e.strerror}")
