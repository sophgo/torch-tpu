import os
import subprocess
import sys
GLOBAL_FAILED = "fail"

def runcmd(command):
    try:
         ret = subprocess.run(command,shell=True, capture_output=True,encoding="utf-8",timeout=1000,check=True)
         if ret.returncode == 0:
            print(ret.stdout)
            return ret.stdout
         else:
            print("Try error:",command, GLOBAL_FAILED)
            return ret.stdout + GLOBAL_FAILED
    except subprocess.CalledProcessError as e:
        print("e.output",e.output)
        return  "Except error:"+command +GLOBAL_FAILED

class Global_Regression_Tester():
    # control top file must be skipped
    # some ops must be skipped
    def filter_skipped_path_utest_new(self):
        self.any_utest_files_list = list(set(self.any_utest_files_list)-set(self.top_python_file_list))
        self.any_utest_files_list = list(set(self.any_utest_files_list)-set(self.global_skip_utest))
        assert len(self.any_utest_files_list)>0, "[ERROR]This assert points that global_skip_utest contains any_utest_files_list, ADD some tests do not belong to global_skip_utest!"

    def filter_skipped_path_old_test(self):
        self.cmp_old_test_files_list = list(set(self.cmp_old_test_files_list )-set(self.skip_old_test))
        assert  len(self.cmp_old_test_files_list)>0, "[ERROR]This assert points that global_skip_utest contains skip_old_test, or python/test has been abondoned!"


    def __init__(self):
        self.control_cmd = "python3  "

        self.chip = os.environ['CHIP_ARCH']
        self.failed_keys = [GLOBAL_FAILED]
        self.any_utest_files_list =  os.listdir("./")
        self.utest_files_list =[]
        self.top_python_file_list = ['top_utest.py', 'utest_cmd.py']
        self.global_skip_utest = ['mlp.py','slice.py',"stack.py"]
        self.filter_skipped_path_utest_new()

        ###[HELPER]You can test specific utest when changing self.any_utest_files_list
        #self.global_skip_utest = []
        #self.any_utest_files_list =["slice.py"]
        self.cmp_old_test_files_list = os.listdir("./../test")
        self.skip_old_test = ['utils.py']
        self.filter_skipped_path_old_test()

        self.dict_error_static = {'f32':[],'f16':[]}

    #this function is voliently searching GLOBAL_FAILED in each utest print info
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

    #This function will print info about what's the files presented in python/test but not in python/utest
    def cmp_old_test(self):
        files_in_newutest_rather_test = []
        files_in_test_rather_newutest = []
        for utest_per in self.utest_files_list:
                new_2_old_str = utest_per.split(self.control_cmd)[1]
                if new_2_old_str not in self.cmp_old_test_files_list:
                    files_in_newutest_rather_test+=[new_2_old_str]

        for old_test_per in self.cmp_old_test_files_list:
            #[Notice]here utest_files_list has been added by "python3 "
            old_2_new_str = self.control_cmd+old_test_per
            if old_2_new_str not in self.utest_files_list:
                    files_in_test_rather_newutest+=[old_test_per]
        print("")
        print("*************CURRENT UTEST SUPPORT INFO LIST ********************")
        print("[WARNING]These files in new utest_ops but not in old test:",files_in_newutest_rather_test )
        print("[MUST-TODO]These files in old test but not tested in new utest_ops:",files_in_test_rather_newutest )
        print("[MUST-TODO]These files might be not created or skipped in list <global_skip_utest> or one utest_new contains multi-test-old" )

    #gen "python x.py"
    def gen_cmd_utest(self):
        assert len(self.utest_files_list)>0
        for id  in range(len(self.utest_files_list)):
            self.utest_files_list[id] =  self.control_cmd + self.utest_files_list[id]

    #check every file is python files
    #filter top_python and skipped_utest again, same as self.filter_skipped_path_utest_new()
    def clean_utest_file_list(self):
        assert len(self.any_utest_files_list)>0
        for each_file_path in self.any_utest_files_list:
            if each_file_path.endswith(".py") and each_file_path not in self.top_python_file_list and each_file_path not in self.global_skip_utest :
                self.utest_files_list +=[each_file_path]
        assert len(self.utest_files_list)>0, "at least test something!"

    def prepare_utests(self):
        self.clean_utest_file_list()
        self.gen_cmd_utest()

    def get_all_utest_result(self):
        self.prepare_utests()

        succeed_result,failed_result = [], []

        for single_utest in self.utest_files_list:
            info = runcmd(single_utest)
            single_utest_name = single_utest.split(self.control_cmd)[1]
            if self.search_failed_info(info):
                failed_result +=[single_utest_name]
            else:
                succeed_result +=[single_utest_name]
            if len(failed_result)>0:
                self.search_stats_error( info, single_utest_name)

        print("*************[CHIP-{}]ALL OUPUTS COMPUTED ********************".format(self.chip))
        print("SUCCESS Cases:", succeed_result)
        print("Failed Cases:", failed_result)
        print("Skipped Cases:", self.global_skip_utest)

       #This function will print info about what's the files presented in python/test but not in python/utest
        self.cmp_old_test()

        #Judger for jenkins
        if (len(failed_result)==0):
            print("*************[CHIP-{}]ALL SUCCESSED AND PASSED ********************".format(self.chip))
            sys.exit(0) #for jenkins check when true, cannot use 1 otherwise exit(1) will corrupt the scripts
        else:
            print("*************[CHIP-{}]ERROR CASES EXISTS ********************".format(self.chip))
            self.dict_error_static
            regression_outerlier = failed_result
            for dtype in self.dict_error_static.keys():
                if len(self.dict_error_static[dtype]) > 0:
                    print("Error {} Cases:".format(dtype), self.dict_error_static[dtype])
                    #don't worry, repeated dtype error ops will just be ignored in set deleting
                    regression_outerlier = list(set(regression_outerlier)- set(self.dict_error_static[dtype]))
                else:
                    print("All {} cases passed".format(dtype))
            if len(regression_outerlier)>0:
                 print("*************[CHIP-{}] SERIOUS ERRORS: ON-LINE REGRESSION IS NOT SAME WITH YOUR HOST********************".format(self.chip))
                 print("On-line Error Cases:",regression_outerlier)
                 print("[NOTE] SG2260 will print dozens of online error due to libsopon&cmodel bugs, but values CMP might be passed")
            sys.exit(255) #for jenkins check when failed

if __name__ == "__main__":
    tester = Global_Regression_Tester()
    tester.get_all_utest_result()