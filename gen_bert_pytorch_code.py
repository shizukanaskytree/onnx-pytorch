import onnx
print(onnx.__version__)

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

from onnx_pytorch import code_gen
code_gen.gen("bertsquad-12.onnx", "./")