
===================
torch
===================

.. list-table::
   :widths: 30 10 65
   :header-rows: 1

   * - PyTorch API
     - Compatibility
     - Mapping
   * - torch.default_generator
     - Y
     - default_generator = torch_tpu.tpu.default_generators
   * - torch.SymInt
     - Y
     -
   * - torch.SymFloat
     - Y
     -
   * - torch.SymBool
     - Y
     -
   * - torch.Tag
     - Y
     -
   * - torch.Tag.name
     - Y
     -
   * - torch.is_tensor
     - Y
     - x = torch.randn((1,32)).tpu().half()
       torch.is_tensor(x_tpu)
   * - torch.is_storage
     - Y
     - x = torch.randn((1,32)).tpu().half()
       torch.is_storage(x_tpu)
   * - torch.is_complex
     - Y
     - x = torch.randn((1,32)).tpu().half()
       torch.is_complex(x_tpu)
   * - torch.is_conj
     - Y
     - x = torch.randn((1,32)).tpu().half()
       torch.is_conj(x_tpu)
   * - torch.is_floating_point
     - Y
     - x = torch.randn((1,32)).tpu().half()
       torch.is_floating_point(x_tpu)
   * - torch.is_nonzero
     - Y
     - x = torch.tensor([1.0]).tpu()
       torch.is_nonzero(x_tpu)
   * - torch.set_default_dtype
     - Y
     - x_tpu = torch.randn((1,32)).to('tpu')
       torch.set_default_dtype(torch.float64)
   * - torch.get_default_dtype
     - Y
     - 
   * - torch.set_default_device
     - Y
     - x = torch.randn((1,32))
       print(x.device)
       torch.set_default_device('tpu')
       x = torch.randn((1,32))
       print(x.device)
   * - torch.set_default_tensor_type
     - Y
     - print(torch.tensor([1.2, 3]).tpu().dtype) 
       torch.set_default_tensor_type(torch.DoubleTensor)
       print(torch.tensor([1.2, 3]).tpu().dtype )
   * - torch.numel
     - Y
     - x_tpu = torch.randn(3,4).tpu()
       torch.numel(x_tpu)
   * - torch.set_printoptions
     - Y
     - 
   * - torch.set_flush_denormal
     - Y
     - 
   * - torch.tensor
     - Y
     - x_tpu = torch.tensor([0,1]).tpu()
   * - torch.sparse_coo_tensor
     - 
     - x_tpu = torch.randn((1,32)).tpu()
       indics = torch.tensor([[0,1], [2,1]]).tpu()
       sparse_tensor = torch.sparse_coo_tensor(indics, x_tpu, [3,3])
	 * - torch.sparse_csr_tensor
     - 
     - values = torch.tensor([1, 2, 3, 4, 5, 6]).tpu()
       crow_indices = torch.tensor([0, 1, 2, 3, 6]).tpu()
       col_indices = torch.tensor([0, 1, 2, 0, 2, 3]).tpu()
       sparse_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(4, 4))
   * - torch.sparse_csc_tensor
     - 
     - ccol_indices = [0, 2, 4]
       row_indices = [0, 1, 0, 1]
       values = [1, 2, 3, 4]
       torch.sparse_csc_tensor(torch.tensor(ccol_indices, dtype=torch.int64).tpu(),torch.tensor(row_indices, dtype=torch.int64).tpu(),torch.tensor(values).tpu(), dtype=torch.double)
   * - torch.sparse_bsr_tensor
     - 
     - crow_indices = [0, 1, 2]
       col_indices = [0, 1]
       values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
       torch.sparse_bsr_tensor(torch.tensor(crow_indices, dtype=torch.int64).tpu(),torch.tensor(col_indices, dtype=torch.int64).tpu(),torch.tensor(values).tpu(), dtype=torch.double)
   * - torch.sparse_bsc_tensor
     - 
     - crow_indices = [0, 1, 2]
       col_indices = [0, 1]
       values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
       torch.sparse_bsc_tensor(torch.tensor(crow_indices, dtype=torch.int64).tpu(),torch.tensor(col_indices, dtype=torch.int64).tpu(),torch.tensor(values).tpu(), dtype=torch.double)
   * - torch.asarray
     - Y
     - a = torch.tensor([1, 2, 3]).tpu()
       b = torch.asarray(a).tpu()
       print(a.data_ptr() == b.data_ptr())
   * - torch.as_tensor
     - Y
     - a = torch.asarray([1, 2, 3]).tpu()
       t = torch.as_tensor(a)
   * - torch.as_strided
     - Y
     - x = torch.randn(3, 3).tpu()
       t = torch.as_strided(x, (2, 2), (1, 2))
   * - torch.from_numpy
     - 
     - a = numpy.array([1, 2, 3])
       t = torch.from_numpy(a)
   * - torch.from_dlpack
     - 
     - import torch.utils.dlpack
       t = torch.arange(4).tpu()
       t2 = torch.from_dlpack(t)
   * - torch.frombuffer
     - 
     - import array
       a = array.array('i', [1, 2, 3]).tpu()
       t = torch.frombuffer(a, dtype=torch.int32)
   * - torch.zeros
     - Y
     - x = torch.zeros((2, 3),device='tpu')
   * - torch.zeros_like
     - Y
     - input = torch.empty(2, 3).tpu()
   * - torch.ones
     - Y
     - x = torch.ones((2, 3),device='tpu')
   * - torch.ones_like
     - Y
     - input = torch.empty(2, 3).tpu()
   * - torch.arange
     - Y
     - x = torch.arange(5,device='tpu')
   * - torch.range
     - 
     - x = torch.range(1,4, device='tpu')
   * - torch.linspace
     - 
     - x = torch.linspace(3, 10, steps=5, device='tpu')
   * - torch.logspace
     - 
     - x = torch.logspace(start=-10, end=10, steps=5, device='tpu')
   * - torch.eye
     - 
     - x = torch.eye(3, device='tpu')
   * - torch.empty
     - Y
     - x = torch.empty((2,3), device='tpu')
   * - torch.empty_like
     - Y
     - y = torch.empty((2,3), device='tpu')
       x = torch.empty_like(y)
   * - torch.empty_strided
     - Y
     - x = torch.empty_strided((2, 3), (1, 2), device='tpu')
   * - torch.full
     - Y
     - x = torch.full((2, 3), 3.141592, device='tpu')
   * - torch.full_like
     - Y
     - y = torch.full((2, 3), 3.141592, device='tpu')
       x = torch.full_like(y, 1)
   * - torch.quantize_per_tensor
     - 
     - y = torch.tensor([-1.0, 0.0, 1.0, 2.0]).tpu()
       x = torch.quantize_per_tensor(y, 0.1, 10, torch.quint8)
   * - torch.quantize_per_channel
     - 
     - y = torch.tensor([[-1.0, 0.0], [1.0, 2.0]]).tpu()
       x = torch.quantize_per_channel(y, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8)
   * - torch.dequantize
     - 
     - float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
       scale = 0.1
       zero_point = 10
       dtype = torch.qint32
       quantized_tensor = torch.quantize_per_tensor(float_tensor, scale, zero_point, dtype)
       torch.dequantize(quantized_tensor.tpu())
   * - torch.complex
     - 
     - real = torch.tensor([1, 2], dtype=torch.float32).tpu()
       imag = torch.tensor([3, 4], dtype=torch.float32).tpu()
       torch.complex(real, imag)
   * - torch.polar
     - 
     - abs = torch.tensor([1, 2], dtype=torch.float64).tpu()
       angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64).tpu()
       torch.polar(abs, angle)
   * - torch.heaviside
     - 
     - input = torch.tensor([-1.5, 0, 2.0]).tpu()
       values = torch.tensor([0.5]).tpu()
       x = torch.heaviside(input, values)
   * - torch.adjoint
     - 
     - x = torch.arange(4, dtype=torch.float)
       A = torch.complex(x, x).reshape(2, 2).contiguous().tpu()
       A.adjoint()
   * - torch.argwhere
     - 
     - t = torch.tensor([1, 0, 1]).tpu()
       torch.argwhere(t)
   * - torch.cat
     - Y
     - x = torch.randn(2, 3).tpu()
       torch.cat((x, x, x), 0)
   * - torch.concat
     - Y
     - x = torch.randn(2, 3).tpu()
       torch.concat((x, x, x), 0)
   * - torch.concatenate
     - Y
     - x = torch.randn(2, 3).tpu()
       torch.concatenate((x, x, x), 0)
   * - torch.conj
     - Y
     - x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]).tpu()
       x.is_conj()
   * - torch.chunk
     - Y
     - x = torch.arange(11).tpu()
       x.chunk(6)
   * - torch.dsplit
     - Y
     - t = torch.arange(16.0).reshape(2, 2, 4).tpu()
       torch.dsplit(t, [3, 6])
   * - torch.column_stack
     - 
     - a = torch.tensor([1, 2, 3]).tpu()
       b = torch.tensor([4, 5, 6]).tpu()
       torch.column_stack((a, b))
   * - torch.dstack
     - 
     - a = torch.tensor([1, 2, 3]).tpu()
       b = torch.tensor([4, 5, 6]).tpu()
       torch.dstack((a,b))
   * - torch.gather
     - 
     - t = torch.tensor([[1, 2], [3, 4]]).tpu()
       z = torch.tensor([[0, 0], [1, 0]]).tpu()
       gather(t, 1, z)
   * - torch.hsplit
     - Y
     - t = torch.arange(16.0).reshape(4,4).tpu()
       torch.hsplit(t, 2)
   * - torch.hstack
     - 
     - a = torch.tensor([1, 2, 3]).tpu()
       b = torch.tensor([4, 5, 6]).tpu()
       torch.hstack((a,b))
   * - torch.index_add
     - 
     - x = torch.ones((5, 3), device='tpu')
       t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float).tpu()
       index = torch.tensor([0, 4, 2]).tpu()
       x.index_add(0, index, t)
   * - torch.index_copy
     - 
     - x = torch.ones((5, 3), device='tpu')
       t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float).tpu()
       index = torch.tensor([0, 4, 2]).tpu()
       x.index_copy(0, index, t)
   * - torch.index_reduce
     - 
     - x = torch.empty(5, 3).fill_(2).tpu()
       t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float).tpu()
       index = torch.tensor([0, 4, 2, 0]).tpu()
       x.index_reduce(0, index, t, 'prod')
   * - torch.index_select
     - Y
     - x = torch.randn(3, 4).tpu()
       indices = torch.tensor([0, 2]).tpu()
       torch.index_select(x, 0, indices)
   * - torch.masked_select
     - 
     - x = torch.randn(3, 4).tpu()
       mask = x.ge(0.5)
       torch.masked_select(x, mask)
   * - torch.movedim
     - Y
     - t = torch.randn(3,2,1).tpu()
       x = torch.movedim(t, 1, 0)
   * - torch.moveaxis
     - Y
     - t = torch.randn(3,2,1).tpu()
       x = torch.moveaxis(t, 1, 0)
   * - torch.narrow
     - Y
     - y = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).tpu()
       x = torch.narrow(y, 0, 0, 2)
   * - torch.narrow_copy
     - Y
     - y = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).tpu()
       x = torch.narrow_copy(y, 0, 0, 2)
   * - torch.nonzero
     - 
     - y = torch.tensor([1, 1, 1, 0, 1]).tpu()
       x = torch.nonzero(y)
   * - torch.permute
     - Y
     - y = torch.randn(2, 3, 5).tpu()
       x = torch.permute(y, (2, 0, 1))
   * - torch.reshape
     - Y
     - torch.arange(16.0).reshape(4,4).tpu()
   * - torch.row_stack
     - 
     - a = torch.tensor([1, 2, 3]).tpu()
       b = torch.tensor([4, 5, 6]).tpu()
       torch.row_stack((a,b))
   * - torch.select
     - 
     - tensor = torch.tensor([[1, 2], [3, 4], [5, 6]]).tpu()
       tensor0 = torch.tensor([1]).tpu()
       selected_row = torch.select(tensor, 0, tensor0)
   * - torch.scatter
     - 
     - src = torch.arange(1, 11).reshape((2, 5)).tpu()
       index = torch.tensor([[0, 1, 2, 0]]).tpu()
       torch.zeros(3, 5, dtype=src.dtype).scatter(0, index, src)
   * - torch.diagonal_scatter
     - Y
     - a = torch.zeros(3, 3).tpu()
       b = torch.ones(3).tpu()
       torch.diagonal_scatter(a, b, 0)
   * - torch.select_scatter
     - Y
     - a = torch.zeros(3, 3).tpu()
       b = torch.ones(3).tpu()
       a.select_scatter(b, 0, 0)
   * - torch.slice_scatter
     - Y
     - a = torch.zeros(8, 8).tpu()
       b = torch.ones(2, 8).tpu()
       a.slice_scatter(b, start=6)
   * - torch.scatter_add
     - 
     - src = torch.ones((2, 5)).tpu()
       index = torch.tensor([[0, 1, 2, 0, 0]]).tpu()
       torch.zeros(3, 5, dtype=src.dtype).scatter_add(0, index, src)
   * - torch.scatter_reduce
     - 
     - src = torch.tensor([1., 2., 3., 4., 5., 6.]).tpu()
       index = torch.tensor([0, 1, 0, 1, 2, 1]).tpu()
       input = torch.tensor([1., 2., 3., 4.]).tpu()
       input.scatter_reduce(0, index, src, reduce="sum")
   * - torch.split
     - Y
     - a = torch.arange(10).reshape(5, 2).tpu()
       torch.split(a, 2)
   * - torch.squeeze
     - Y
     - x = torch.zeros(2, 1, 2, 1, 2).tpu()
       print(x.cpu().size())
       y = torch.squeeze(x)
       print(y.cpu())
   * - torch.stack
     - 
     - tensor1 = torch.tensor([1, 2, 3]).tpu()
       tensor2 = torch.tensor([4, 5, 6]).tpu()
       x = torch.stack((tensor1, tensor2), dim=0)
   * - torch.swapaxes
     - 
     - y = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]]).tpu()
       torch.swapaxes(y, 0, 1)
   * - torch.swapdims
     - 
     - y = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]]).tpu()
       torch.swapdims(y, 0, 1)
   * - torch.t
     - Y
     - y = torch.randn(2, 3).tpu()
       torch.t(y)
   * - torch.take
     - Y
     - src = torch.tensor([[4, 3, 5],[6, 7, 8]]).tpu()
       index = torch.tensor([0, 2, 5]).tpu()
       torch.take(src, index)
   * - torch.take_along_dim
     - 
     - t = torch.tensor([[10, 30, 20], [60, 40, 50]]).tpu()
       max_idx = torch.argmax(t)
       torch.take_along_dim(t, max_idx)
   * - torch.tensor_split
     - Y
     - y = torch.arange(8).tpu()
       torch.tensor_split(y, 3)
   * - torch.tile
     - Y
     - y = torch.tensor([1, 2, 3]).tpu()
       x = y.tile((2,))
   * - torch.transpose
     - Y
     - y = torch.randn(2, 3).tpu()
       torch.transpose(y, 0, 1)
   * - torch.unbind
     - Y
     - y = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).tpu()
       torch.unbind(y)
   * - torch.unsqueeze
     - Y
     - y = torch.tensor([1, 2, 3, 4]).tpu()
       torch.unsqueeze(y, 0)
   * - torch.vsplit
     - Y
     - t = torch.arange(16.0).reshape(4,4).tpu()
       torch.vsplit(t, 2)
   * - torch.vstack
     - 
     - a = torch.tensor([1, 2, 3]).tpu()
       b = torch.tensor([4, 5, 6]).tpu()
       torch.vstack((a,b))
   * - torch.where
     - Y
     - x = torch.randn(3, 2).tpu()
       torch.where(x > 0, 1.0, 0.0)
   * - torch.Generator
     - Y
     - g_tpu = torch.Generator(device='tpu')
   * - torch.Generator.device
     - Y
     - g_tpu = torch.Generator(device='tpu')
       g_tpu.device
   * - torch.Generator.get_state
     - Y
     - g_tpu = torch.Generator(device='tpu')
       g_tpu.get_state()
   * - torch.Generator.initial_seed
     - Y
     - g_tpu = torch.Generator(device='tpu')
       g_tpu.initial_seed()
   * - torch.Generator.manual_seed
     - Y
     - g_tpu = torch.Generator(device='tpu')
       g_tpu.manual_seed(2147483647)
   * - torch.Generator.seed
     - Y
     - g_tpu = torch.Generator(device='tpu')
       g_tpu.seed()
   * - torch.Generator.set_state
     - Y
     - g_tpu = torch.Generator(device='tpu')
       g_tpu_other = torch.Generator(device='tpu')
       g_tpu.set_state(g_tpu_other.get_state())
   * - torch.seed
     - 
     - 
   * - torch.manual_seed
     - 
     - 
   * - torch.initial_seed
     - 
     - 
   * - torch.get_rng_state
     - Y
     - torch.get_rng_state().tpu()
   * - torch.set_rng_state
     - Y
     - saved_rng_state = torch.get_rng_state().tpu()
       torch.set_rng_state(saved_rng_state)
   * - torch.bernoulli
     - 
     - a = torch.ones(3, 3).tpu()
       torch.bernoulli(a)
   * - torch.multinomial
     - 
     - weights = torch.tensor([0, 10, 3, 0], dtype=torch.float).tpu()
       torch.multinomial(weights, 2)
   * - torch.normal
     - 
     - a = torch.arange(1., 11.).tpu()
       b = torch.arange(1, 0, -0.1).tpu()
       torch.normal(mean=a, std=b)
   * - torch.poisson
     - 
     - rates = torch.rand(4, 4) * 5
       torch.poisson(rates.tpu())
   * - torch.rand
     - Y
     - torch.rand(4, device='tpu')
   * - torch.rand_like
     - Y
     - original_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]).tpu()
       torch.rand_like(original_tensor, dtype=torch.float)
   * - torch.randint
     - Y
     - torch.randint(3, 5, (3,),device='tpu')
   * - torch.randint_like
     - Y
     - original_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]).tpu()
       torch.randint_like(original_tensor, low=0, high=10)
   * - torch.randn
     - Y
     - torch.randn(4, device='tpu')
   * - torch.randn_like
     - Y
     - original_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).tpu()
       torch.randn_like(original_tensor)
   * - torch.quasirandom.SobolEngine
     - 
     - 
   * - torch.quasirandom.SobolEngine.draw
     - 
     - 
   * - torch.quasirandom.SobolEngine.draw_base2
     - 
     - 
   * - torch.quasirandom.SobolEngine.fast_forward
     - 
     - 
   * - torch.quasirandom.SobolEngine.reset
     - 
     - 
   * - torch.save
     - Y
     - 
   * - torch.load
     - Y
     - 
   * - torch.get_num_threads
     - Y
     - 
   * - torch.set_num_threads
     - Y
     - 
   * - torch.get_num_interop_threads
     - Y
     - 
   * - torch.set_num_interop_threads
     - Y
     - 
   * - torch.no_grad
     - Y
     - x = torch.tensor([1.], requires_grad=True).tpu()
       with torch.no_grad():
           y = x * 2
       print( y.requires_grad)
   * - torch.enable_grad
     - Y
     - x = torch.tensor([1.], requires_grad=True).tpu()
       with torch.no_grad():
           with torch.enable_grad():
           y = x * 2
       print( y.requires_grad)
   * - torch.set_grad_enabled
     - Y
     - 
   * - torch.is_grad_enabled
     - Y
     - 
   * - torch.inference_mode
     - Y
     - x = torch.ones(1, 2, 3, requires_grad=True).tpu()
       with torch.inference_mode():
           y = x * x
       print(y.requires_grad)
   * - torch.is_inference_mode_enabled
     - Y
     - 
   * - torch.abs
     - 
     - y = torch.tensor([-1, -2, 3]).tpu()
       torch.abs(y)
   * - torch.absolute
     - 
     - y = torch.tensor([-1, -2, 3]).tpu()
       torch.absolute(y)
   * - torch.acos
     - Y
     - y = torch.randn(4).tpu()
       torch.acos(y)
   * - torch.arccos
     - Y
     - y = torch.randn(4).tpu()
       torch.arccos(y)
   * - torch.acosh
     - Y
     - y = torch.randn(4).tpu()
       torch.acosh(y)
   * - torch.arccosh
     - Y
     - y = torch.randn(4).tpu()
       torch.arccosh(y)
   * - torch.add
     - Y
     - y = torch.randn(4).tpu()
       torch.add(y, 20)
   * - torch.addcdiv
     - 
     - t = torch.randn(1, 3).tpu()
       t1 = torch.randn(3, 1).tpu()
       t2 = torch.randn(1, 3).tpu()
       torch.addcdiv(t, t1, t2, value=0.1)
   * - torch.addcmul
     - Y
     - t = torch.randn(1, 3).tpu()
       t1 = torch.randn(3, 1).tpu()
       t2 = torch.randn(1, 3).tpu()
       x = torch.addcmul(t, t1, t2, value=0.1)
   * - torch.angle
     - 
     - y = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]).tpu()
       torch.angle(y)*180/3.14159
   * - torch.asin
     - Y
     - y = torch.randn(4).tpu()
       torch.asin(y)
   * - torch.arcsin
     - Y
     - y = torch.randn(4).tpu()
       torch.arcsin(y)