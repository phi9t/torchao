import rich
import rich.pretty
import rich.traceback
import torch
import torch._dynamo
import torchao
from torchao.quantization import DEFAULT_INT4_AUTOQUANT_CLASS_LIST


# TorchRuntimeError: Failed running call_module fn_0(*(FakeTensor(..., device='cuda:0', size=(32, 32), dtype=torch.bfloat16),), **{}):
# 'FakeTensor' object has no attribute 'int_data'
#
# from user code:
#    File ".../torch/_dynamo/external_utils.py", line 38, in inner
#     return fn(*args, **kwargs)
#
# Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
#
# You can suppress this exception and fall back to eager by setting:
#     import torch._dynamo
#     torch._dynamo.config.suppress_errors = True
torch._dynamo.config.suppress_errors = True

rich.pretty.install()
rich.traceback.install(show_locals=False, extra_lines=4)

# https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#autoquantization
ex_model = torch.nn.Sequential(torch.nn.Linear(32, 64)).cuda().to(torch.bfloat16)
ex_model_input = torch.randn(32, 32, dtype=torch.bfloat16, device="cuda")
expected_output = ex_model(ex_model_input)

# ex_model_quantized = torchao.autoquant(torch.compile(ex_model, mode="max-autotune"))
ex_model_quantized = torchao.autoquant(
    torch.compile(ex_model, mode="max-autotune"),
    qtensor_class_list=DEFAULT_INT4_AUTOQUANT_CLASS_LIST,
)

fetechd_output = ex_model_quantized(ex_model_input)
torch.testing.assert_allclose(fetechd_output, expected_output, rtol=1e-3, atol=1e-3)
