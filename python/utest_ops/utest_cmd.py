import os
import subprocess

def runcmd(command):
    ret = subprocess.run(command,shell=True, capture_output=True,encoding="utf-8",timeout=100,check=True)

    if ret.returncode == 0:
        print(ret.stdout)
    else:
        print("error:",command, "failed!")
    return ret.stdout

def search_failed_info(info):
    failed_keys = ["fail" ]
    info = info.lower()
    for i in failed_keys:
        if i in info:
            print("[Watchout] failure because key- {}!".format(i))
            return 1
    return 0

def get_all_utest_result():
    any_test_files_list =  os.listdir("./")
    utest_files_list =[]
    top_py_file_list = ['top_utest.py', 'utest_cmd.py']
    for each_file_path in any_test_files_list:
        if each_file_path.endswith(".py") and each_file_path not in top_py_file_list :
            utest_files_list +=[each_file_path]

    cmd = "python3  "
    for id  in range(len(utest_files_list)):
        utest_files_list[id] =  cmd + utest_files_list[id]
    
    succeed_result = []
    failed_result =[]
    all_num_utest = len(utest_files_list)
    for i in utest_files_list:
        info = runcmd(i)
        if search_failed_info(info):
            failed_result +=[i.split(cmd)[1]]
        else:
            succeed_result +=[i.split(cmd)[1]]
    print("*************ALL OUPUTS COMPUTED ********************")
    print("SUCCESS Cases:", succeed_result)
    print("Failed Cases:", failed_result)
    if (len(failed_result)>0):
            print("*************EXIST ERROR!********************")
            return 1
    if (len(succeed_result)==all_num_utest):
            print("*************ALL SUCCESSED AND PASSED ********************")
            return 0

get_all_utest_result()