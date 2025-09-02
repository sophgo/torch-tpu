#!/bin/bash

# 用途：Torch-TPU 开源至 Github 时使用，Github 不支持大量lfs文件，需要转换成常规文件
# 脚本功能：将 Git LFS 管理的 .so 文件转换为常规 Git 管理的文件
# 使用方法：在 Git 仓库根目录下执行此脚本
#           使用此脚本会创建新分支并将 lfs 文件转换为常规文件
#           转换完毕后可将此分支 force push 到 GitHub
#           push 完毕后直接切回其他分支，此分支删除即可

# 检查是否在 Git 仓库中
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "错误：当前目录不是 Git 仓库"
    exit 1
fi

# 检查是否安装了 Git LFS
if ! git lfs --version > /dev/null 2>&1; then
    echo "错误：Git LFS 未安装"
    echo "请先安装 Git LFS: https://git-lfs.com/"
    exit 1
fi

# 备份当前分支
current_branch=$(git branch --show-current)
backup_branch="none-lfs-branch-$(date +%s)"
echo "创建非LFS分支: $backup_branch"
git checkout -b "$backup_branch"

# 1. 找出所有 LFS 管理的 .so 文件
echo "查找所有 LFS 管理的 .so 文件..."
lfs_files=$(git lfs ls-files | awk '{print $3}')

if [ -z "$lfs_files" ]; then
    echo "没有找到 LFS 管理的 .so 文件"
    exit 0
fi

echo "找到以下 LFS 管理的 .so 文件:"
echo "$lfs_files"

# 2. 从 LFS 中移除这些文件
echo "从 LFS 中移除这些文件..."
git lfs uninstall

# 3. 删除并重新添加这些文件
for file in $lfs_files; do
    echo "处理文件: $file"

    # 确保文件存在
    if [ ! -f "$file" ]; then
        echo "警告: 文件 $file 不存在，跳过"
        continue
    fi

    # 备份文件
    cp "$file" "$file.bak"

    # 从 Git 和 LFS 中删除
    git rm --cached "$file"

    # 恢复文件
    mv "$file.bak" "$file"

done

echo "转换完成！"
echo "请检查更改并推送到远程仓库"

