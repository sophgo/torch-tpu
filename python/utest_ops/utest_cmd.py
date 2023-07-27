import os
import subprocess
import sys

def runcmd(command):
    try:
         ret = subprocess.run(command,shell=True, capture_output=True,encoding="utf-8",timeout=1000,check=True)
         if ret.returncode == 0:
            print(ret.stdout)
         else:
            print("error:",command, "failed!")
         return ret.stdout
    except subprocess.CalledProcessError as e:
        print(e.output)
        return  "error:"+command +"failed!"

class Global_Regression_Tester():
    def __init__(self):

        self.chip = os.environ['CHIP_ARCH']
        self.failed_keys = ["fail" ]
        self.any_test_files_list =  os.listdir("./")
        self.utest_files_list =[]
        self.top_py_file_list = ['top_utest.py', 'utest_cmd.py']
        self.global_skip_test = ['mlp.py','slice.py']

        self.dict_error_static = {'f32':[],'f16':[]}
    def search_failed_info(self, info):
        info = info.lower()
        for i in self.failed_keys:
            if i in info:
                print("[Watchout] failure because key-'{}' matches info!".format(i))
                return 1
        return 0

    def search_stats_error(self, info, single_utest_name):
        for dtype in self.dict_error_static.keys():
            if "dtype {} exist errors".format(dtype) in info:
                self.dict_error_static[dtype] +=[single_utest_name]

    def get_all_utest_result(self):
        for each_file_path in self.any_test_files_list:
            if each_file_path.endswith(".py") and each_file_path not in self.top_py_file_list and each_file_path not in self.global_skip_test :
                self.utest_files_list +=[each_file_path]

        cmd = "python3  "
        for id  in range(len(self.utest_files_list)):
            self.utest_files_list[id] =  cmd + self.utest_files_list[id]

        succeed_result = []
        failed_result =[]
        all_num_utest = len(self.utest_files_list)

        for single_utest in self.utest_files_list:
            info = runcmd(single_utest)
            single_utest_name = single_utest.split(cmd)[1]
            if self.search_failed_info(info):
                failed_result +=[single_utest_name]
            else:
                succeed_result +=[single_utest_name]
            if len(failed_result)>0:
                self.search_stats_error( info, single_utest_name)

        print("*************[CHIP-{}]ALL OUPUTS COMPUTED ********************".format(self.chip))
        print("SUCCESS Cases:", succeed_result)
        print("Failed Cases:", failed_result)
        print("Skipped Cases:", self.global_skip_test)

        #Judger for jenkins
        if (len(failed_result)==0):
            print("*************[CHIP-{}]ALL SUCCESSED AND PASSED ********************".format(self.chip))
            sys.exit(0) #for jenkins check when true, cannot use 1 otherwise exit(1) will corrupt the scripts
        else:
            print("*************[CHIP-{}]ERROR CASES EXISTS ********************".format(self.chip))
            self.dict_error_static
            for dtype in self.dict_error_static.keys():
                if len(self.dict_error_static[dtype]) > 0:
                    print("Error {} Cases:".format(dtype), self.dict_error_static[dtype])
                else:
                    print("All {} cases passed".format(dtype))
            sys.exit(255) #for jenkins check when failed

if __name__ == "__main__":
    tester = Global_Regression_Tester()
    tester.get_all_utest_result()