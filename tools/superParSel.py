import json
import os
import argparse
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
import plotly.graph_objects as go
from functools import reduce
import re

# GB = 1000 ** 3  # replace 1<<30
# TB = 1000 ** 4  # replace 1<<40
GB = 1<<30
TB = 1<<40
US = 1e6

reqLatency = 75
tpuMem = 128

def roundup(x, div):
    return div*np.ceil(x/div)

def rsum(*numbers):
    return reduce(lambda s,x: s + (1./x if x!=0 else 0), numbers, 0)

def btoi(abool: bool) -> int:
    return 1 if abool else 0

def addSelfPoint(s:str) -> str:
    if isinstance(s, str) and s[:5]!="self.":
        pattern = r"(\w+)(.?)"
        matches = re.findall(pattern, s)
        return "".join( f"self.{m[0]}{m[1]}" if not m[0].isdigit() else m[0]+m[1] for m in matches )
    else:
        return s

def isProper(x):
    if isinstance(x, (int, float)):
        return x>0
    else:
        return x

MatrixProps = ('N', 'C', 'H', 'W')
MatrixPropsOrders = {s:i for i, s in enumerate(MatrixProps)}
ico_list = ["in0", "in1", "out"]


class InitializationError(Exception):
    pass

class ProcessError(Exception):
    pass

class LLM:
    Qtype = 'f16'
    name = None

    debug = False
    mode = 'infer'
    
    batch = 8
    tp = 1
    pp = 1
    nparams = 0
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isProper(v):
                setattr(self, k, v)

        modelJson = getattr(self, "modelJson", None)
        if not modelJson:
            if os.path.exists(f"{self.name}.json"):
                modelJson = f"{self.name}.json"
            else:
                raise InitializationError(f"Please input the json file of {self.name}.")
        
        tpuJson = getattr(self, "tpuJson", None)
        if not tpuJson:
            raise InitializationError(f"Please input the json file of device.")

        if not self._initModel(modelJson, tpuJson):
            raise InitializationError("Reading model fail!")

        if not self._checkAttr():
            raise InitializationError("Model initialize fail!")

        print(f"INFO: {self.name} generate successfully!\n")

    def getName(self) -> str:
        return self.name
    
    def getNParam(self):
        return self.nparams

    def _usePredefinedModel(self) -> None:
        preDefinedModel = {
            "LLaMA-2-7B":{
                "layer_num": 32,
                "intermediate_size": 11008,
                "head": 32,
                "kv_heads": 32,
            },
            "LLaMA-2-13B":{
                "layer_num": 40,
                "intermediate_size": 13824,
                "head": 40,
                "kv_heads": 40,
            },
            "LLaMA-2-70B":{
                "layer_num": 80,
                "intermediate_size": 28672,
                "head": 64,
                "kv_heads": 8,
            }
        }
        if self.name in preDefinedModel:
            for k, v in preDefinedModel[self.name].items():
                setattr(self, k, v)

    def _checkAttr(self) -> bool:
        check_items = ('name', 'layer_num', 'intermediate_size', 'head', 'kv_heads')
        for k in check_items:
            if not hasattr(self, k):
                print(f"{k} not in init parameters!")
                return False
            
        mode_choices = ('infer', 'backward', 'train', 'prefill')
        if self.mode not in mode_choices:
            raise InitializationError(f"{self.mode} not defined, select in {mode_choices}.")
        
        return True
    
    def _initModel(self, modelJson, tpuJson) -> bool:
        if not os.path.exists(modelJson):
            modelJson = 'examples/'+modelJson
            if not os.path.exists(modelJson):
                print(modelJson + ' not exist!')
                return False
        
        if not os.path.exists(tpuJson):
            print(tpuJson + ' not exist!')
            return False
        print(f'Reading initial model from {modelJson}...')

        with open(tpuJson) as f:
            for k, v in json.load(f).items():
                if hasattr(self, k) and isProper(getattr(self, k)):
                    print(f"WARN: {k} has been overwriten to be {getattr(self, k)}")
                    continue
                setattr(self, k, v)

        with open(modelJson) as f:
            for k, v in json.load(f).items():
                if hasattr(self, k) and isProper(getattr(self, k)):
                    print(f"WARN: {k} has been overwriten to be {getattr(self, k)}")
                    continue
                setattr(self, k, v)
                
        if not hasattr(self, "layers"):
            print("INFO: No layers specified, reading LLaMA layer as default.")
            with open("examples/LLaMA_layers.json") as f:
                self.layers = json.load(f)

        self.hidden_size = self.head * self.block_size
        self.T_us = self.compute * TB / US #(self.compute<<10) * 1e3
        self.timeScale    = US/self.DRAMBW/GB
        self.calTimeScale = US/self.compute/TB /self.core_num
        if self.mode in ('train', 'prefill'):
            print(f"Note: {self.mode}'s seq_len will be {self.token_size}")
            self.seq_len = self.token_size
        else:
            self.seq_len = 1
        
        self._fullfillModule(self.layers)
        
        if self.debug:
            with open("test.json", "w") as f:
                json.dump(self.layers, f, indent=4)

        if self.mode == "train" and self.Qtype in ('f8', 'w4a16'):
            raise ProcessError("Temporary training cant use f8 or w4a16.")
        return True
    
    def _fullfillModule(self, d:dict):
        for k, v in d.items():
            if isinstance(v, dict):
                for s in MatrixProps + ("w4a16", "f8", "f16"):
                    if s in d and s not in v:
                        v[s] = d[s]

                if "matrix" in d and "matrix" in v and d["matrix"].get("common", True):
                    for kk in d["matrix"]:
                        if kk not in v["matrix"]:
                            v["matrix"][kk] = d["matrix"][kk]

                ### tmp use only for True case
                for s in ("IM", "OM"):
                    if d.get(s, False) and "type" in v and s not in v:
                        v[s] = False

                self._fullfillModule(v)

            elif isinstance(v, str):
                if k in MatrixProps:
                    d[k] = self._getValue(addSelfPoint(v))
            
            elif isinstance(v, list):
                for i, s in enumerate(v):
                    if isinstance(s, str):
                        v[i] = self._getValue(addSelfPoint(s))
                    elif s==-1:
                        if MatrixProps[i] in d:
                            v[i] = d[MatrixProps[i]]
                        else:
                            print(f"Error! {MatrixProps[i]} has not been defined, plz check.")
    
    def getTokenSChip(self, tp:int, batch:int, Qtype:str=None, latency:float=-1)    -> float:
        if latency > 0:
            return self.seq_len * batch/latency*1e3 / tp
        else:
            return self.seq_len * batch/self.getLatency(tp, batch, Qtype)*1e3 / tp

    def _setSuperPars(self, tp:int, batch:int, Qtype:str=None) -> None:
        self.tp = tp
        self.batch = batch
        self.Qtype = Qtype

    def getLatency(self, tp:int, batch:int, Qtype:str=None) -> float:
        self._setSuperPars(tp, batch, Qtype)
        modules = self.layers

        total_time = np.zeros(2) # 2: not, inLayer
        self.firstInfer_k = None
        if self.mode != "backward":
            for k, v in modules.items():
                if isinstance(v, dict):
                    self.getNParamFlag = self.layer_num if v.get("inLayer", False) else 1
                    if self.firstInfer_k is None:
                        self.firstInfer_k = k
                    # X_l+1 = X_l @ W_l^T  ~~>  out = in0 @ in1  ~~>  [a,c] ~ [a,b] @ [b,c]  ~~> FLOPs: 2abc  ~~> IOsize: ab+bc+ac  !!!! be careful if MM is in a fusion operator 
                    tmpTime = self._getTime(v)
                    total_time[btoi(v.get("inLayer", False))] += tmpTime
                    if self.debug:
                        print(f"alg infer    time[us] {k}: {tmpTime:.2f}")
        self.getNParamFlag = 0

        if self.mode in ("backward", "train"):
            for k, v in reversed(modules.items()):
                if isinstance(v, dict):
                    # GradX_l-1 = GradX_l @ W       ~~>  in0 ~ out @ in1^T  ~~>  [a,b] ~ [a,c] @ [c,b]  ~~> FLOPs: 2abc  ~~> IOsize: ab+bc+ac
                    if k != self.firstInfer_k: # No previous layer
                        t_GradX = self._getTime(v, backtype='X')
                    # GradW_l^T = X_l-1^T @ GradX_l ~~>  in1 ~ in0^T @ out  ~~>  [b,c] ~ [b,a] @ [a,c]  ~~> FLOPs: 2abc  ~~> IOsize: ab+bc+ac
                    t_GradW = self._getTime(v, backtype='W')
                    total_time[btoi(v.get("inLayer", False))] += t_GradX + t_GradW
                    if self.debug:
                        print(f"alg backX    time[us] {k}: {t_GradX:.2f}")
                        print(f"alg backW    time[us] {k}: {t_GradW:.2f}")
                        
        total_time[1] *= self.layer_num
        total_time /= 1e3 # to ms

        self.latency = total_time.sum() #ms
        if self.mode in ("backward", "train"):
            ### update params
            t_upd = 2*self.getNParam()*self.timeScale/1e3 # to ms
            self.latency += t_upd
            if self.debug:
                print(f"alg parameters update time [ms]: {t_upd:.2f}")

        if self.debug:
            with open("test_time.json", "w") as f:
                json.dump(self.layers, f, indent=4)
        print()
        return self.latency
    
    def getMemoryTotal(self, tp:int, batch:int, Qtype:str=None) -> float:
        return np.sum(self.getMemory(tp,batch,Qtype))

    def getMemory(self, tp:int, batch:int, Qtype:str=None) -> np.ndarray:
        self._setSuperPars(tp, batch, Qtype)
        modules = self.layers

        mem = np.zeros((2,3)) # 2: not, inLayer; 3: cache/act, weight, other
        for k, v in modules.items():
            if isinstance(v, dict):
                tmpMem = self._getMemory(v)
                mem[btoi(v.get("inLayer", False))] += tmpMem
                if self.debug:
                    print(f"alg debug memory [MB] {k}: {self._getMemory(v)}")
        mem /= 1024 # to GB
        if self.mode == "train":
            mem[..., 1] *= 8 # each training param -> 16 Bytes 

        total_mem = np.zeros(3)
        if self.pp <= 2:
            mem[1] *= self.layer_num
            total_mem = np.sum(mem, axis=0)/self.pp
        else:
            minMem = 1e9
            memIO = mem[0].sum()/2
            memSL = mem[1].sum() 
            n_io = 0
            ioGT = False
            for n in range(self.layer_num):
                tot_mid = self.layer_num-2*n
                if tot_mid<0 or tot_mid%(self.pp-2)!=0:
                    continue
                tmpMem = max( memIO + memSL*n, memSL*(tot_mid//(self.pp-2)) )
                if tmpMem < minMem:
                    minMem = tmpMem
                    n_io = n
                    ioGT = (memIO + memSL*n) > (memSL*(tot_mid//(self.pp-2)))
                else:
                    break
            print(f"INFO: Auto select {n_io} layers with I/O embedding.")
            total_mem = mem[0]/2 + mem[1]*n_io if ioGT else mem[1]*( (self.layer_num-2*n_io)//(self.pp-2) ) 
        
        if self.debug:
            with open("test_memory.json", "w") as f:
                json.dump(self.layers, f, indent=4)
        print()
        return total_mem

    def _getMemory(self, d:dict, keyname:str=None, backtype:str=None) -> np.ndarray:
        mem = np.zeros(3) # cache/activations, weight, other
        for k, v in d.items():
            if isinstance(v, dict) and k!="matrix" and k[:2]!="t_":
                mem += self._getMemory(v, k, backtype)
        
        tmpMem = self._getAlgDataBytes(d, True, keyname, backtype)/(1<<20) # MB
        mem += tmpMem
        if self.debug:
            d["mem"] = tmpMem.tolist()
        return mem

    def _getTime(self, d:dict, keyname:str=None, backtype:str=None) -> float:
        tot = 0
        for k, v in d.items():
            if isinstance(v, dict) and k!="matrix" and k[:2]!="t_":
                tot += self._getTime(v, k, backtype)

        t_dma = self._getAlgDataBytes(d, False, keyname, backtype).sum()*self.timeScale
        t_tiu = self._getCalTime(d, keyname, backtype)
        t_use = max(t_dma, t_tiu)
        tot += t_use
        if self.debug and t_use>1:
            tmpstr = 'infer' if backtype is None else f'back{backtype}'
            d[f"t_dma_{tmpstr}"] = int(t_dma)
            d[f"t_tiu_{tmpstr}"] = int(t_tiu)

        return tot
    
    def _getCalTime(self, alg:dict, keyname:str=None, backtype:str=None) -> float:
        if alg.get("noCalTime", False):
            return 0
        
        ### ATT_QKV's backward, get Q,K,V and dY(gradY), out dQ,dK,dV
        ### S = Q@K^T, dS = dY@V^T, dQ = dS@K, dK = dS^T@Q, dV = S^T@dY
        if keyname == "input":
            if backtype=="X":
                ### 3*Q@K^T + 2*S@V
                return 3*self._getCalTime(self.layers["ATT_QKV"]["Score"]) + 2*self._getCalTime(self.layers["ATT_QKV"]["MatMul"])
            else:
                return 0
        
        if "type" not in alg:
            return 0
        t  = alg["type"]
        
        if "matrix" not in alg:
            return 0
        matrix = alg["matrix"]
        f8    = self.Qtype=='f8' and isinstance(matrix, dict) and matrix.get("f8",   False)
        hasBias =                    isinstance(matrix, dict) and matrix.get("bias", False)

        if t in ("MM2", "MM2_NT"):
            if "out" not in matrix or "in1" not in matrix:
                raise ProcessError(f"No output&in1 info in {matrix}")
            loop_i = -1
            tmpmaxS = 1
            for i in (1,2,0): # C H N
                if self._getValue(matrix["out"][i]) > tmpmaxS:
                    loop_i = i
                    tmpmaxS = self._getValue(matrix["out"][i])
                    
            if loop_i < 0:
                loop_i = 1
            
            flops  = 2*reduce(lambda x,y: x*y, (self._getValue(matrix["out"][i]) if i!=loop_i else roundup(self._getValue(matrix["out"][i]), self.NPU_NUM) for i in range(len(MatrixProps))))
            tmpS = 'H' if t[-1]!='T' else 'W'
            flops *= self._getValue(matrix["in1"][MatrixPropsOrders[tmpS]])

            if hasBias:
                flops += reduce(lambda x,y: x*y, (self._getValue(x) for x in matrix['out']))
            calT = flops*self.calTimeScale
            # if self.mode == "train" and alg.get("isParam", True):
            #     calT *= 3 # forward 1 + backward 2
            return calT if not f8 else f8/2
        
        elif t in ("Mix", "AR", "CDMA", "Act"):
            return 0
        else:
            raise ProcessError(f"Unexpected type {t}")

    def _getAlgDataBytes(self, alg:dict, memOpt:bool=False, keyname:str=None, backtype:str=None):
        ds = np.zeros(3)

        ### ATT_QKV's input
        if keyname == "input":
            if backtype=="X":
                ### get Q,K,V and dY(gradY), out dQ,dK,dV
                ds[0] = self.getMatrixDataBytes(alg, tuple(alg.keys()) + ('Q', 'K', 'V'))
            elif backtype=="W":
                pass
            else:
                ds[0] = self.getMatrixDataBytes(alg, ('Q', 'K', 'V'))
            return ds
        
        ### no KV cache in train/prefill
        if keyname == "cache":
            if self.mode not in ('train', 'prefill', 'backward'):
                ds[0] = self.getMatrixDataBytes(alg)
            return ds

        nm = alg.get("NM", False)
        if nm:
            return ds
        
        if "matrix" not in alg:
            return ds
        matrix = alg["matrix"]
        
        # im = alg.get("IM", True) # input
        # cm = alg.get("CM", True) # calculation
        # om = alg.get("OM", True) # output
        ico_m = [ alg.get(s, True) for s in ("IM", "CM", "OM")]
        if "type" not in alg and sum(ico_m)>0:
            if self.mode != "train":
                ds[2] = self.getMatrixDataBytes(matrix)
            return ds
        
        t  = alg["type"]
        if memOpt and t in ('AR', ):
            return ds
        
        isParam = alg.get("isParam", True)
        s_noMem   = alg.get("s_noMem", None)
        if isParam and self.getNParamFlag>0 and (t[:3] == "MM2" or t=="Mix"):
            self.nparams += self.getNParamFlag*self.getMatrixDataBytes(matrix, ("in1",))

        if t in ("MM2", "Mix", "AR", "MM2_NT"):
            rw_list = ico_list.copy()
            if memOpt and s_noMem:
                rw_list.remove(s_noMem)
            
            if backtype is not None:
                if isParam:
                    ds[0]  = self.getMatrixDataBytes(matrix, ("in0",))
                    ds[1]  = self.getMatrixDataBytes(matrix, ("in1",))
                    ds[0] += self.getMatrixDataBytes(matrix, ("out",))
                return ds

            if memOpt and self.mode in ("train", "backward") and (t[:3] == "MM2" or t=="Mix") and isParam:
                for i, s in enumerate(ico_list):
                    if i==2 and not alg.get("saveOutput", False):
                        continue
                    ds[i] = self.getMatrixDataBytes(matrix, (s,))
                return ds
            else:
                for i, s in enumerate(ico_list):
                    if not ico_m[i]:
                        rw_list.remove(s)
                    elif i==1:
                        ds[i] = self.getMatrixDataBytes(matrix, (s,))
                        rw_list.remove(s)

            if rw_list:
                ds[2] = self.getMatrixDataBytes(matrix, rw_list)

        elif t == "CDMA": # AllReduce
            if not memOpt:
                ds[2] = self.getAllReduceTime(self.getMatrixDataBytes(matrix))/self.timeScale

        else:
            print(alg)
            print("Error! Not a defined calculation, plz check.")
            ds = np.ones_like(ds)*1e9

        return ds

    def getMatrixDataBytes(self, matrix:list|dict, keys:list|tuple=None):
        if isinstance(matrix, list):
            tmplen = 1 if keys is None else len(keys)
            return self.data_byte * tmplen * reduce(lambda x,y: x*y, (self._getValue(s) for s in matrix))
        
        elif isinstance(matrix, dict):
            hasBias = matrix.get("bias", False)
            keys = matrix.keys() if keys is None else keys

            ds = 0
            for s in keys:
                if s not in matrix:
                    raise ProcessError(f"Calculation key {s} from {keys} not in {matrix}!")
                elif isinstance(matrix[s], list):
                    scale = self.data_byte
                    if self.Qtype == "w4a16":
                        if s == "in1": #  w 1/2 + scale 2./gs + zp 1./gs/2
                            scale = (1./2 + self.data_byte/self.group_size + 1./self.group_size/2) 
                    elif self.Qtype == "f8":
                        if s in ("in0", "in1"): # in/w  1 + ignore scale 
                            scale = 1.

                    ds += scale * reduce(lambda x,y: x*y, (self._getValue(x) for x in matrix[s]))
                    if hasBias and s=="in1":
                        ds += self.getOutFeature(matrix[s]) # Bias size too small to be calculated
            return ds
        else:
            raise ProcessError(f"Unknown matrix dtype! {matrix}")
    
    def getOutFeature(self, dims:list):
        # Usually the second non-one dim
        outfeature = self._getValue(dims[-1])
        skipDim = True
        for dim in dims:
            if self._getValue(dim)!=1:
                if not skipDim:
                    outfeature = self._getValue(dim)
                skipDim = False
        return outfeature

    def _getValue(self, s):
        num =  eval(s) if isinstance(s, str) else s
        return np.ceil(num)
    
    def getMatrixProp(self, m:dict, s:str):
        v = m[s] if s in m else m[s.split('_')[-1]]
        if isinstance(v, str):
            v = eval(v)
        return v

    def getAllReduceTime(self, db) -> float:
        tp = self.tp
        if tp == 1:
            return 0
        GBus = 1e-3 #US/GB
        data_GBus_per_tp = db/tp*GBus
        if tp == 2:
            return          data_GBus_per_tp * rsum(self.v_CDMA,   self.v_ddr2l2,   self.v_SUM,   self.v_l22ddr) + self.lt_fire + self.lt_sync*2
        else:
            t0 = (tp/2-1) *(data_GBus_per_tp * rsum(self.v_CDMA,   self.v_ddr2l2,   self.v_SUM,   self.v_l22ddr) + self.lt_fire + self.lt_sync*4)
            t1 =            data_GBus_per_tp * rsum(self.v_CDMA, 2*self.v_ddr2l2, 2*self.v_SUM, 2*self.v_l22ddr)
            t2 = (tp/2-1) *(data_GBus_per_tp * rsum(self.v_CDMA) )
            return t0 + t1 + t2
        
    def dump(self, outf="network.json") -> None:
        print("INFO: Print network info...")
        ### todo: Only remain shape and time, mem info
        # tmpJson = {}
        # for k, v in self.layers.items():

        print("INFO: Done.")
        

def draw3D(m:LLM, flag="tps", Qtype='w4a16', tps=2**np.arange(10), batches=2**np.arange(10)):
    Qtypes = ('w4a16', 'f16', 'f8')
    if Qtype not in Qtypes:
        print(f"Error: Qtype={Qtype}. Select Qtype in {Qtypes}.")
        return
    
    flags = ('tps', 'mem')
    if flag not in flags:
        print(f"Error! flag={flag}, select in {flags}.")
        return
  
    x, y = np.meshgrid(tps, batches)

    latency = np.zeros_like(x, dtype=float)
    tokenSChip = np.zeros_like(x, dtype=float)
    mem = np.zeros_like(x, dtype=float)
    # ==========Gen point (latency<75)==========
    highlight_points=[]
    maxValue = 100
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            latency[i, j]  = m.getLatency(x[i,j], y[i,j], Qtype)
            tokenSChip[i,j]=m.getTokenSChip(x[i,j], y[i,j], Qtype, latency[i, j])
            mem[i,j] = m.getMemoryTotal(x[i,j], y[i,j], Qtype)
            if mem[i,j] < tpuMem:
                if flag == 'tps' and tokenSChip[i, j] > maxValue:
                    maxValue = tokenSChip[i, j]

                if (flag == 'tps' and latency[i, j] < reqLatency) or (flag == 'mem' and mem[i,j]<tpuMem):
                    highlight_points.append((x[i,j], y[i,j]))
                
    z1 = tokenSChip
    z2 = 1e3*y/x/reqLatency
    highlight_str = f'latency<{reqLatency}ms'
    if flag == 'mem':
        z1 = mem
        z2 = tpuMem*np.ones_like(z1)
        highlight_str = f'memory<{tpuMem}GB'
        maxValue = 520


    print(f"============ {Qtype} {highlight_str} configs (tp,batch) ============")
    print(highlight_points)

    # ==========Plot==========
    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z1, colorscale='Blues'))
    fig.add_trace(go.Surface(x=x, y=y, z=z2, colorscale=[[0, 'orange'], [1, 'orange']], opacity=0.4, showscale=False, name='Plane'))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if mem[i,j]<tpuMem or flag=='mem': # only mark memory under tpuMem
                fig.add_trace(go.Scatter3d(
                    x=[x[i, j]], y=[y[i, j]], z=[z1[i, j]],
                    mode='text',
                    text=[f'{z1[i, j]:.2f}'],
                    textposition='top center',
                    textfont=dict(size=13, color='black')
                ))

    for xi in np.unique(x):
        yi = y[:, 0]
        zi = z1[:, np.where(x[0] == xi)[0][0]]
        fig.add_trace(go.Scatter3d(
            x=[xi]*len(yi), y=yi, z=zi, mode='lines', name=f'X={xi}',line=dict(color='white', width=3),
        ))
    for yi in np.unique(y):
        xi = x[0, :]
        zi = z1[np.where(y[:, 0] == yi)[0][0], :]
        fig.add_trace(go.Scatter3d(
            x=xi, y=[yi]*len(xi), z=zi, mode='lines', name=f'Y={yi}',line=dict(color='white', width=3),
        ))

    for point in highlight_points:
        xi, yi = point
        zi = z1[np.where(y[:, 0] == yi)[0][0], np.where(x[0] == xi)[0][0]]
        fig.add_trace(go.Scatter3d(
            x=[xi], y=[yi], z=[zi],
            mode='markers',
            marker=dict(color='purple', size=8),
            name=f'Highlight ({xi}, {yi})'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                type='log',
                tickvals=tps,
                ticktext=tps,
                title='TP',
                showgrid=True,
                gridcolor='lightgrey',
                gridwidth=1
            ),
            yaxis=dict(
                type='log',
                tickvals=batches,
                ticktext=batches,
                title='Batch',
                showgrid=True,
                gridcolor='lightgrey',
                gridwidth=1
            ),
            zaxis=dict(
                title='token/s/chip' if flag=='tps' else 'Memory / GB',
                showgrid=True,
                gridcolor='lightgrey',
                gridwidth=1,
                range = [0,maxValue],
                tickvals=np.linspace(0, maxValue, 5),
                ticktext=np.linspace(0, maxValue, 5)
            )
        )
    )


    # ==========Save Fig==========
    fig.write_html(f"{m.getName()}_{Qtype}_{flag}.html")
    pass

def drawTimeLine(m:LLM, Qtype):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print(f"Now, drawing time consume lines for {m.mode} name={m.getName()}, batch={m.batch}, tp={m.tp}, Qtype={Qtype}.")
    plt.figure(figsize=(12, 6))
    
    tot_time = m.latency*1e3
    inLayerScale = m.layer_num
    def plot_segment(name: str, d: dict, start=0, height=0, scale=1., label_lower=False) -> float:        
        if "inLayer" in d:
            scale = inLayerScale
        
        delatT = 0
        cur_lable_lower = label_lower
        for k, v in d.items():
            if isinstance(v, dict):
                tmpT = plot_segment(k, v, start+delatT, height+1, scale, cur_lable_lower)
                delatT += tmpT
                if tmpT > 3e-2:
                    cur_lable_lower = not cur_lable_lower

        if "t_io" in d:
            delatT += scale * max(d["t_io"], d["t_cal"])/tot_time

        if delatT<1e-6:
            return 0

        if delatT > 3e-2:
            plt.text(start, height-0.3-0.1*label_lower, name)
            plt.text(start, height-0.18-0.06*label_lower, f"{tot_time*delatT/1e3:.2f}")
        plt.plot([start, start+delatT], [height, height], linewidth=10, solid_capstyle='butt')
        return delatT

    plot_segment(f"{m.name} {m.mode}: {m.tp} tp, {m.batch} batch, quantization={Qtype}. Time unit: [ms]", m.layers)

    
    plt.axis('off')
    plt.savefig(f"{m.getName()}_{m.mode}_tp{m.tp}_batch{m.batch}_{Qtype}.png")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='Latency Calculator.')
    parser.add_argument('--modelJson', default='examples/LLaMA-3-405B.json', type=str, help='Your model json file.')
    parser.add_argument('--tpuJson', default='examples/tpu.json', type=str, help='Your tpu info json.')
    parser.add_argument('--Qtype', default='f16', type=str, help='Your quantization type.')
    parser.add_argument('--debug', default=0, type=int, help='Debug mode for simple check.')
    parser.add_argument('--mode', default='infer', type=str, choices=('infer', 'train', 'prefill'), help='Training, Inference or prefill.')
    parser.add_argument('--reqLatency', default=75, type=int, help='The latency requirement (ms).')
    parser.add_argument('--reqMemory', default=128, type=int, help='The maximum memory in one device (GB).')
    
    parser.add_argument('--token_size', default=-1, type=int, help='The token size of LLM. (Will overwrite the property from network json)')

    parser.add_argument('--tp', default=-1, type=int, help='The number of device.')
    parser.add_argument('--pp', default=1, type=int, help='The pipeline parallelism.')
    parser.add_argument('--batch', default=-1, type=int, help='Batch.')
    args = parser.parse_args()

    reqLatency = args.reqLatency
    tpuMem = args.reqMemory

    m = LLM(**vars(args))

    if args.tp>0 and args.batch>0:
        m.debug = True
        print(f"Debug run for specific config, tp={args.tp}, batch={args.batch}, quantization type={args.Qtype}.")

        latency = m.getLatency(args.tp, args.batch, args.Qtype)
        memDetail = np.round(m.getMemory(args.tp, args.batch, args.Qtype),2)
        memStr0 = "KVcache" if args.mode != "train" else "Activation"
        print(f"{m.mode} latency = {latency:.2f} ms\ntotal memory usage = {memDetail.sum():.2f} GB\n[{memStr0}, weight, other] = {memDetail}GB\ntps = {m.getTokenSChip(args.tp, args.batch, args.Qtype, latency):.2f}")
        drawTimeLine(m, args.Qtype)
        print("Network total parameter bytes =", m.getNParam()/GB, "GB")
        exit()
    
    if args.modelJson == "all" and args.tpuJson is not None:
        from glob import glob
        for jsonFile in glob("examples/*.json"):
            with open(jsonFile) as f:
                jsondata = json.load(f)
                if "superParSel" in jsondata and "name" in jsondata:
                    if args.Qtype == "all":
                        for s in ('f16', 'w4a16', 'f8'):
                            for p in ('tps', 'mem'):
                                draw3D(m, p, args.Qtype)
                    else:
                        for p in ('tps', 'mem'):
                            draw3D(m, p, args.Qtype)

    else:
        for p in ('tps', 'mem'):
            draw3D(m, p, args.Qtype)
        

