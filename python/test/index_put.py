import torch
import torch_tpu

torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"
# max_record = int(1e6)
# torch.ops.my_ops.enable_profile(max_record, 2)

def case1():
    '''multi-dim index_put - basic execution test'''
    self = torch.randn(64, 3, 20, 20, 85)
    value = torch.randn(3238, 85)
    i0 = torch.randint(0, 64, (3238,), dtype=torch.int32)
    i1 = torch.randint(0, 3, (3238,), dtype=torch.int32)
    i2 = torch.randint(0, 20, (3238,), dtype=torch.int32)
    i3 = torch.randint(0, 20, (3238,), dtype=torch.int32)
    
    self_tpu = self.clone().to(device)
    pos = (i0[0].item(), i1[0].item(), i2[0].item(), i3[0].item())
    before = self_tpu[pos].clone()
    
    self_tpu[i0.to(device), i1.to(device), i2.to(device), i3.to(device)] = value.to(device)
    
    after = self_tpu[pos]
    print(f"case1: executed={not torch.allclose(before.cpu(), after.cpu())}")

def case2():
    '''index_put: replace & accumulate'''
    # replace
    self = torch.zeros(5).to(device)
    idx = torch.tensor([1, 3], dtype=torch.int32).to(device)
    val = torch.tensor([10.0, 20.0]).to(device)
    self[idx] = val
    print("case2 replace:", torch.allclose(self.cpu(), torch.tensor([0., 10., 0., 20., 0.])))
    
    # accumulate
    self = torch.ones(5).to(device)
    self.index_put_([idx], val, accumulate=True)
    print("case2 accumulate:", torch.allclose(self.cpu(), torch.tensor([1., 11., 1., 21., 1.])))

def case3():
    '''large scale multi-dim'''
    cases = [(64,3,20,20,85,3238), (64,3,40,40,85,3969), (64,3,80,80,85,2974)]
    for (d0,d1,d2,d3,d4,n) in cases:
        self = torch.randn(d0,d1,d2,d3,d4).to(device)
        val = torch.randn(n,d4).to(device)
        i0 = torch.randint(0,d0,(n,),dtype=torch.int32).to(device)
        i1 = torch.randint(0,d1,(n,),dtype=torch.int32).to(device)
        i2 = torch.randint(0,d2,(n,),dtype=torch.int32).to(device)
        i3 = torch.randint(0,d3,(n,),dtype=torch.int32).to(device)
        
        pos = (i0[0].item(), i1[0].item(), i2[0].item(), i3[0].item())
        before = self[pos].clone()
        self[i0,i1,i2,i3] = val
        print(f"case3 {(d0,d1,d2,d3,d4),n}: executed={not torch.allclose(before.cpu(), self[pos].cpu())}")

def case4():
    '''partial indexing'''
    self = torch.zeros(5,3,4,4,10).to(device)
    i0 = torch.tensor([0,1,2], dtype=torch.int32).to(device)
    i1 = torch.tensor([0,1,2], dtype=torch.int32).to(device)
    i2 = torch.tensor([0,1,2], dtype=torch.int32).to(device)
    i3 = torch.tensor([0,1,2], dtype=torch.int32).to(device)
    val = (torch.arange(30, dtype=torch.float32).reshape(3,10) + 100).to(device)
    
    self[i0,i1,i2,i3] = val
    match = torch.allclose(self[0,0,0,0].cpu(), val[0].cpu())
    print(f"case4: {match}")

def case5():
    '''bool indices'''
    self = torch.zeros(5).to(device)
    mask = torch.tensor([True,False,True,False,False]).to(device)
    val = torch.tensor([1.0,2.0]).to(device)
    self[mask] = val
    print(f"case5: {self.cpu()}")

def case6():
    '''None indices - comprehensive testing of CPU fallback'''
    # Test 1: Leading None indices
    try:
        self = torch.zeros(3,4,5).to(device)
        idx = torch.tensor([0,1], dtype=torch.int32).to(device)
        val = torch.ones(2,4,5).to(device)
        self[None,None,idx] = val
        
        # Verify with CPU reference
        self_cpu = torch.zeros(3,4,5)
        idx_cpu = torch.tensor([0,1], dtype=torch.int32)
        val_cpu = torch.ones(2,4,5)
        self_cpu[None,None,idx_cpu] = val_cpu
        
        match = torch.allclose(self.cpu(), self_cpu)
        print(f"case6-none-leading: {'PASS' if match else 'FAIL'}")
    except Exception as e:
        print(f"case6-none-leading: FAIL - {str(e)[:100]}")
    
    # Test 2: Middle None indices
    try:
        self = torch.zeros(3,4,5).to(device)
        idx0 = torch.tensor([0,1], dtype=torch.int32).to(device)
        idx1 = torch.tensor([2,3], dtype=torch.int32).to(device)
        val = torch.ones(2,1,5).to(device) * 200  # Shape: [2, 1, 5]
        self[idx0, None, idx1] = val
        
        # Verify with CPU reference
        self_cpu = torch.zeros(3,4,5)
        idx0_cpu = torch.tensor([0,1], dtype=torch.int32)
        idx1_cpu = torch.tensor([2,3], dtype=torch.int32)
        val_cpu = torch.ones(2,1,5) * 200
        self_cpu[idx0_cpu, None, idx1_cpu] = val_cpu
        
        match = torch.allclose(self.cpu(), self_cpu)
        print(f"case6-none-middle: {'PASS' if match else 'FAIL'}")
    except Exception as e:
        print(f"case6-none-middle: FAIL - {str(e)[:100]}")
    
    # Test 3: Trailing None indices
    try:
        self = torch.zeros(3,4,5).to(device)
        idx0 = torch.tensor([0,1], dtype=torch.int32).to(device)
        idx1 = torch.tensor([2,3], dtype=torch.int32).to(device)
        val = torch.ones(2,1,5).to(device) * 300  # Shape: [2, 1, 5]
        self[idx0, idx1, None] = val
        
        # Verify with CPU reference
        self_cpu = torch.zeros(3,4,5)
        idx0_cpu = torch.tensor([0,1], dtype=torch.int32)
        idx1_cpu = torch.tensor([2,3], dtype=torch.int32)
        val_cpu = torch.ones(2,1,5) * 300
        self_cpu[idx0_cpu, idx1_cpu, None] = val_cpu
        
        match = torch.allclose(self.cpu(), self_cpu)
        print(f"case6-none-trailing: {'PASS' if match else 'FAIL'}")
    except Exception as e:
        print(f"case6-none-trailing: FAIL - {str(e)[:100]}")
    
    # Test 4: Multiple None indices
    try:
        self = torch.zeros(3,4,5).to(device)
        idx = torch.tensor([0,1], dtype=torch.int32).to(device)
        val = torch.ones(1,2,1,4,5).to(device) * 400  # Shape: [1, 2, 1, 4, 5]
        self[None, idx, None] = val
        
        # Verify with CPU reference
        self_cpu = torch.zeros(3,4,5)
        idx_cpu = torch.tensor([0,1], dtype=torch.int32)
        val_cpu = torch.ones(1,2,1,4,5) * 400
        self_cpu[None, idx_cpu, None] = val_cpu
        
        match = torch.allclose(self.cpu(), self_cpu)
        print(f"case6-none-multiple: {'PASS' if match else 'FAIL'}")
    except Exception as e:
        print(f"case6-none-multiple: FAIL - {str(e)[:100]}")
    
    # Test 5: Normal indexing for comparison (should use TPU)
    try:
        self = torch.zeros(3,4,5).to(device)
        idx0 = torch.tensor([0,1], dtype=torch.int32).to(device)
        idx1 = torch.tensor([2,3], dtype=torch.int32).to(device)
        val = torch.ones(2,5).to(device) * 500
        self[idx0,idx1] = val
        
        # Compare with CPU reference
        self_cpu = torch.zeros(3,4,5)
        idx0_cpu = torch.tensor([0,1], dtype=torch.int32)
        idx1_cpu = torch.tensor([2,3], dtype=torch.int32)
        val_cpu = torch.ones(2,5) * 500
        self_cpu[idx0_cpu,idx1_cpu] = val_cpu
        
        match = torch.allclose(self.cpu(), self_cpu)
        print(f"case6-normal: {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"    TPU result sum: {self.cpu().sum()}, CPU result sum: {self_cpu.sum()}")
    except Exception as e:
        print(f"case6-normal: FAIL - {str(e)[:100]}")
    
    # Test 6: Mixed tensor index and Python integer (original bug case)
    try:
        self = torch.zeros(5, 6).to(device)
        idx = torch.tensor([1, 2, 3], dtype=torch.int32).to(device)
        val = torch.tensor([10.0, 20.0, 30.0]).to(device)
        self[idx, 4] = val  # Mixed: tensor index + Python int
        
        # Compare with CPU reference
        self_cpu = torch.zeros(5, 6)
        idx_cpu = torch.tensor([1, 2, 3], dtype=torch.int32)
        val_cpu = torch.tensor([10.0, 20.0, 30.0])
        self_cpu[idx_cpu, 4] = val_cpu
        
        match = torch.allclose(self.cpu(), self_cpu)
        print(f"case6-mixed-int: {'PASS' if match else 'FAIL'} (CPU fallback for non-contiguous view)")
        if not match:
            print(f"    TPU result[:, 4]: {self.cpu()[:, 4]}")
            print(f"    CPU result[:, 4]: {self_cpu[:, 4]}")
    except Exception as e:
        print(f"case6-mixed-int: FAIL - {str(e)[:100]}")
    
    # Test 7: None with Python integers (original bug case)
    try:
        self = torch.zeros(5, 6).to(device)
        val = 99.0
        self[None, 3, 4] = val  # None + Python ints
        
        # Compare with CPU reference
        self_cpu = torch.zeros(5, 6)
        self_cpu[None, 3, 4] = val
        
        match = torch.allclose(self.cpu(), self_cpu)
        print(f"case6-none-int: {'PASS' if match else 'FAIL'} (CPU fallback for None indices)")
        if not match:
            print(f"    TPU result[3, 4]: {self.cpu()[3, 4].item()}")
            print(f"    CPU result[3, 4]: {self_cpu[3, 4].item()}")
    except Exception as e:
        print(f"case6-none-int: FAIL - {str(e)[:100]}")

def case7():
    '''various shapes'''
    for size in [5,10,100]:
        self = torch.zeros(size).to(device)
        idx = torch.randint(0,size,(3,),dtype=torch.int32).to(device)
        val = torch.randn(3).to(device)
        self[idx] = val
        print(f"case7-1D {size}: OK")
    
    for h,w in [(3,4),(10,5)]:
        self = torch.zeros(h,w).to(device)
        i0 = torch.randint(0,h,(2,),dtype=torch.int32).to(device)
        i1 = torch.randint(0,w,(2,),dtype=torch.int32).to(device)
        val = torch.randn(2).to(device)
        self[i0,i1] = val
        print(f"case7-2D {h}x{w}: OK")

def case8():
    '''accumulate mode'''
    self = torch.zeros(5).to(device)
    idx = torch.tensor([1,1,1], dtype=torch.int32).to(device)
    val = torch.tensor([1.0,2.0,3.0]).to(device)
    self.index_put_([idx], val, accumulate=True)
    print(f"case8: sum={self[1].item()} (expect 6.0)")

def case9():
    '''boundary conditions'''
    # empty
    self = torch.zeros(5).to(device)
    self[torch.tensor([],dtype=torch.int32).to(device)] = torch.tensor([]).to(device)
    print("case9-empty: OK")
    
    # scalar broadcast
    self = torch.zeros(3).to(device)
    self[torch.tensor([0,2],dtype=torch.int32).to(device)] = 99.0
    print(f"case9-scalar: {self.cpu()}")

def case10():
    '''stress test'''
    self = torch.zeros(10000).to(device)
    idx = torch.randint(0,10000,(1000,),dtype=torch.int32).to(device)
    val = torch.randn(1000).to(device)
    self[idx] = val
    print("case10: OK")

def case11():
    '''comprehensive validation'''
    # basic
    self = torch.zeros(5,3).to(device)
    idx = torch.tensor([0,2,4], dtype=torch.int32).to(device)
    val = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float32).to(device)
    self[idx] = val
    print(f"case11-basic: {torch.allclose(self[0].cpu(), val[0].cpu())}")
    
    # accumulate
    self = torch.ones(3).to(device)
    idx = torch.tensor([1,1,1], dtype=torch.int32).to(device)
    val = torch.tensor([1.0,2.0,3.0]).to(device)
    self.index_put_([idx], val, accumulate=True)
    print(f"case11-accum: {self[1].item()} (expect 7.0)")

def case12():
    '''error handling'''
    # multi-dim index tensor
    try:
        self = torch.zeros(5,5).to(device)
        idx = torch.tensor([[0,1],[2,3]], dtype=torch.int32).to(device)
        self[idx] = torch.ones(2,2).to(device)
        print("case12-multidim: FAIL")
    except RuntimeError as e:
        print(f"case12-multidim: PASS ({'multi-dimensional' in str(e)})")
    
    # bool index (should work)
    self = torch.zeros(5).to(device)
    mask = torch.tensor([True,False,True,False,True]).to(device)
    self[mask] = torch.ones(3).to(device)
    print(f"case12-bool: {self.cpu()}")
    
    # multi-dim index with 2D tensor (uses CPU fallback)
    try:
        self = torch.zeros(5,5,5).to(device)
        i0 = torch.tensor([[0,1],[2,3]], dtype=torch.int32).to(device)
        i1 = torch.tensor([0,1], dtype=torch.int32).to(device)
        value = torch.ones(2,2,5).to(device)
        self[i0,i1] = value
        
        # Verify with CPU reference
        self_cpu = torch.zeros(5,5,5)
        i0_cpu = torch.tensor([[0,1],[2,3]], dtype=torch.int32)
        i1_cpu = torch.tensor([0,1], dtype=torch.int32)
        value_cpu = torch.ones(2,2,5)
        self_cpu[i0_cpu,i1_cpu] = value_cpu
        
        match = torch.allclose(self.cpu(), self_cpu)
        print(f"case12-2D: {'PASS' if match else 'FAIL'} (CPU fallback for 2D index)")
    except RuntimeError as e:
        print(f"case12-2D: UNEXPECTED ERROR - {str(e)[:80]}")

if __name__ == "__main__":
    case1()  # multi-dim basic test
    case2()  # replace & accumulate
    case3()  # large scale
    case4()  # partial indexing
    case5()  # bool indices
    case6()  # leading None error
    case7()  # various shapes
    case8()  # accumulate mode
    case9()  # boundary conditions
    case10() # stress test
    case11() # comprehensive
    case12() # error handling
    # torch.ops.my_ops.disable_profile()
