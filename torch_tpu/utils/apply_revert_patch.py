import os
import sys
import logging
import pkg_resources

patch_log = logging.getLogger(__name__)
# set time and highlight the log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

check_file_exist = lambda file_path: os.path.exists(file_path)
PATCH_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../demo/patch")
patch_log.info(f"Patch directory: {PATCH_DIR}")
directory_stack  = []

apply_table = [
    ["transformers"   , "4.41.2" , "transformers-Sophgo"        ],
    ["accelerate"     , "0.30.1" , "accelerate-Sophgo"    ],
    # ["llamafactory"   , "0.8.3"  , "LLaMA-Factory-Sophgo" ]
]

def pushd(path):
    patch_log.info(f"Pushing {path} to directory stack")
    directory_stack.append(os.getcwd())
    os.chdir(path)

def popd():
    if not directory_stack:
        patch_log.error("Directory stack is empty")
        return
    prev_dir = directory_stack.pop()
    os.chdir(prev_dir)
    patch_log.info(f"Poping {prev_dir} from directory stack")

def _os_system_(cmd: str, save_log: bool = False):
    cmd_str = cmd
    patch_log.info("[Running]: %s", cmd_str)
    ret = os.system(cmd_str)
    if ret == 0:
        patch_log.info("[Success]: %s", cmd_str)
        return True
    else:
        raise RuntimeError("[!Error]: {}".format(cmd_str))
    return False

def check_version(package_name, version):
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
        return installed_version == version
    except pkg_resources.DistributionNotFound:
        patch_log.error(f"Package {package_name} is not installed")
        return False
    except pkg_resources.VersionConflict as e:
        patch_log.error(f"Package {package_name} has version conflict: {e}, we need version {version}")
        return False
    return False

get_package_dir = lambda package_name: pkg_resources.get_distribution(package_name).location

def check_is_applied(package_name):
    package_dir = get_package_dir(package_name) + "/" + package_name
    return check_file_exist(package_dir+"/sophgo.py")

def apply_certain_patch(package_name, patch_name):
    patch_log.info(f"Applying patch {patch_name}")
    if check_is_applied(package_name):
        patch_log.info(f"Patch {patch_name} has been applied")
        return
    package_dir = get_package_dir(package_name) + "/" + package_name
    pushd(package_dir)
    patch_file = os.path.join(PATCH_DIR, f"{patch_name}.patch")
    if not check_file_exist(patch_file):
        patch_log.error(f"Patch file {patch_file} does not exist")
        return
    patch_log.info(f"Try to apply patch {patch_name}")
    res = _os_system_(f"patch -p3  --dry-run < {patch_file}")
    if res:
        patch_log.info(f"Patch {patch_name} apply")
        _os_system_(f"patch -p3 < {patch_file}")
    popd()

def revert_certain_patch(package_name, patch_name):
    patch_log.info(f"Reverting patch {patch_name}")
    if not check_is_applied(package_name):
        patch_log.info(f"Do nothing! Patch {package_name} has not applied!")
        return
    package_dir = get_package_dir(package_name) + "/" + package_name
    pushd(package_dir)
    patch_file = os.path.join(PATCH_DIR, f"{patch_name}.patch")
    if not check_file_exist(patch_file):
        patch_log.error(f"Patch file {patch_file} does not exist")
        return
    _os_system_(f"patch -p3 -R --force < {patch_file}")
    popd()

def apply_all_patches():
    for package_name, version, patch_name in apply_table:
        if not check_version(package_name, version):
            patch_log.error(f"Package {package_name} is not in version {version}")
            return
        apply_certain_patch(package_name, patch_name)

def revert_all_patches():
    for package_name, version, patch_name in apply_table:
        if not check_version(package_name, version):
            patch_log.error(f"Package {package_name} is not in version {version}")
            return
        revert_certain_patch(package_name, patch_name)

# apply_all_patches()
# revert_all_patches()