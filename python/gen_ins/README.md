`gen_ins` folder is only perpose for dump the calculate operation of pytorch trian(or infer)'s process.

- utils.py     
  save the ENVRION to control cmdoel's behavior
  - FORBID_CMD_EXECUTE : '1', 禁止计算，仅dump。'0'，开启计算。
  - FILE_DUMP_CMD : dump出指令文件的名称
  - CMODEL_GLOBAL_MEM_SIZE : GLOABL_MEM的大小。   

  use hack mechanism to insert 'dump ins' operation.     
  - ForwardHack
  - BackwardHack

- gpt3block_TP_fp16.py  
  save GPT3 one block's forward and backward process op's ins.

  usage:
   ``` shell
   $ mkdir ins
   $ cd ins
   $ python ../gpt3block_TP16_fb.py
   ```   