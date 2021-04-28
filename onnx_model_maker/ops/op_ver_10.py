# Autogenerated by onnx-model-maker. Don't modify it manually.

import onnx
import onnx.helper
import onnx.numpy_helper
from onnx_model_maker import omm
from onnx_model_maker import onnx_mm_export
from onnx_model_maker.ops.op_helper import _add_input


@onnx_mm_export("v10.RoiAlign")
def RoiAlign(X, rois, batch_indices, **kwargs):
  _inputs = []
  for i in (X, rois, batch_indices):
    _add_input(i, _inputs)

  idx = omm.op_counter["RoiAlign"]
  omm.op_counter["RoiAlign"] += 1
  node = onnx.helper.make_node("RoiAlign",
                               _inputs, [f"_t_RoiAlign_{idx}"],
                               name=f"RoiAlign_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.ReverseSequence")
def ReverseSequence(input, sequence_lens, **kwargs):
  _inputs = []
  for i in (input, sequence_lens):
    _add_input(i, _inputs)

  idx = omm.op_counter["ReverseSequence"]
  omm.op_counter["ReverseSequence"] += 1
  node = onnx.helper.make_node("ReverseSequence",
                               _inputs, [f"_t_ReverseSequence_{idx}"],
                               name=f"ReverseSequence_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.NonMaxSuppression")
def NonMaxSuppression(boxes, scores, max_output_boxes_per_class=None, iou_threshold=None, score_threshold=None, **kwargs):
  _inputs = []
  for i in (boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
    _add_input(i, _inputs)

  idx = omm.op_counter["NonMaxSuppression"]
  omm.op_counter["NonMaxSuppression"] += 1
  node = onnx.helper.make_node("NonMaxSuppression",
                               _inputs, [f"_t_NonMaxSuppression_{idx}"],
                               name=f"NonMaxSuppression_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.IsInf")
def IsInf(X, **kwargs):
  _inputs = []
  for i in (X, ):
    _add_input(i, _inputs)

  idx = omm.op_counter["IsInf"]
  omm.op_counter["IsInf"] += 1
  node = onnx.helper.make_node("IsInf",
                               _inputs, [f"_t_IsInf_{idx}"],
                               name=f"IsInf_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.QuantizeLinear")
def QuantizeLinear(x, y_scale, y_zero_point=None, **kwargs):
  _inputs = []
  for i in (x, y_scale, y_zero_point):
    _add_input(i, _inputs)

  idx = omm.op_counter["QuantizeLinear"]
  omm.op_counter["QuantizeLinear"] += 1
  node = onnx.helper.make_node("QuantizeLinear",
                               _inputs, [f"_t_QuantizeLinear_{idx}"],
                               name=f"QuantizeLinear_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.QLinearConv")
def QLinearConv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B=None, **kwargs):
  _inputs = []
  for i in (x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B):
    _add_input(i, _inputs)

  idx = omm.op_counter["QLinearConv"]
  omm.op_counter["QLinearConv"] += 1
  node = onnx.helper.make_node("QLinearConv",
                               _inputs, [f"_t_QLinearConv_{idx}"],
                               name=f"QLinearConv_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.ConvInteger")
def ConvInteger(x, w, x_zero_point=None, w_zero_point=None, **kwargs):
  _inputs = []
  for i in (x, w, x_zero_point, w_zero_point):
    _add_input(i, _inputs)

  idx = omm.op_counter["ConvInteger"]
  omm.op_counter["ConvInteger"] += 1
  node = onnx.helper.make_node("ConvInteger",
                               _inputs, [f"_t_ConvInteger_{idx}"],
                               name=f"ConvInteger_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.QLinearMatMul")
def QLinearMatMul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point, **kwargs):
  _inputs = []
  for i in (a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point):
    _add_input(i, _inputs)

  idx = omm.op_counter["QLinearMatMul"]
  omm.op_counter["QLinearMatMul"] += 1
  node = onnx.helper.make_node("QLinearMatMul",
                               _inputs, [f"_t_QLinearMatMul_{idx}"],
                               name=f"QLinearMatMul_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.MatMulInteger")
def MatMulInteger(A, B, a_zero_point=None, b_zero_point=None, **kwargs):
  _inputs = []
  for i in (A, B, a_zero_point, b_zero_point):
    _add_input(i, _inputs)

  idx = omm.op_counter["MatMulInteger"]
  omm.op_counter["MatMulInteger"] += 1
  node = onnx.helper.make_node("MatMulInteger",
                               _inputs, [f"_t_MatMulInteger_{idx}"],
                               name=f"MatMulInteger_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.StringNormalizer")
def StringNormalizer(X, **kwargs):
  _inputs = []
  for i in (X, ):
    _add_input(i, _inputs)

  idx = omm.op_counter["StringNormalizer"]
  omm.op_counter["StringNormalizer"] += 1
  node = onnx.helper.make_node("StringNormalizer",
                               _inputs, [f"_t_StringNormalizer_{idx}"],
                               name=f"StringNormalizer_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.Mod")
def Mod(A, B, **kwargs):
  _inputs = []
  for i in (A, B):
    _add_input(i, _inputs)

  idx = omm.op_counter["Mod"]
  omm.op_counter["Mod"] += 1
  node = onnx.helper.make_node("Mod",
                               _inputs, [f"_t_Mod_{idx}"],
                               name=f"Mod_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.DequantizeLinear")
def DequantizeLinear(x, x_scale, x_zero_point=None, **kwargs):
  _inputs = []
  for i in (x, x_scale, x_zero_point):
    _add_input(i, _inputs)

  idx = omm.op_counter["DequantizeLinear"]
  omm.op_counter["DequantizeLinear"] += 1
  node = onnx.helper.make_node("DequantizeLinear",
                               _inputs, [f"_t_DequantizeLinear_{idx}"],
                               name=f"DequantizeLinear_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.ThresholdedRelu")
def ThresholdedRelu(X, **kwargs):
  _inputs = []
  for i in (X, ):
    _add_input(i, _inputs)

  idx = omm.op_counter["ThresholdedRelu"]
  omm.op_counter["ThresholdedRelu"] += 1
  node = onnx.helper.make_node("ThresholdedRelu",
                               _inputs, [f"_t_ThresholdedRelu_{idx}"],
                               name=f"ThresholdedRelu_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.Upsample")
def Upsample(X, scales, **kwargs):
  _inputs = []
  for i in (X, scales):
    _add_input(i, _inputs)

  idx = omm.op_counter["Upsample"]
  omm.op_counter["Upsample"] += 1
  node = onnx.helper.make_node("Upsample",
                               _inputs, [f"_t_Upsample_{idx}"],
                               name=f"Upsample_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.Slice")
def Slice(data, starts, ends, axes=None, steps=None, **kwargs):
  _inputs = []
  for i in (data, starts, ends, axes, steps):
    _add_input(i, _inputs)

  idx = omm.op_counter["Slice"]
  omm.op_counter["Slice"] += 1
  node = onnx.helper.make_node("Slice",
                               _inputs, [f"_t_Slice_{idx}"],
                               name=f"Slice_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.TopK")
def TopK(X, K, **kwargs):
  _inputs = []
  for i in (X, K):
    _add_input(i, _inputs)

  idx = omm.op_counter["TopK"]
  omm.op_counter["TopK"] += 1
  node = onnx.helper.make_node("TopK",
                               _inputs, [f"_t_TopK_{idx}"],
                               name=f"TopK_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.Resize")
def Resize(X, scales, **kwargs):
  _inputs = []
  for i in (X, scales):
    _add_input(i, _inputs)

  idx = omm.op_counter["Resize"]
  omm.op_counter["Resize"] += 1
  node = onnx.helper.make_node("Resize",
                               _inputs, [f"_t_Resize_{idx}"],
                               name=f"Resize_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.MaxPool")
def MaxPool(X, **kwargs):
  _inputs = []
  for i in (X, ):
    _add_input(i, _inputs)

  idx = omm.op_counter["MaxPool"]
  omm.op_counter["MaxPool"] += 1
  node = onnx.helper.make_node("MaxPool",
                               _inputs, [f"_t_MaxPool_{idx}"],
                               name=f"MaxPool_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.Dropout")
def Dropout(data, **kwargs):
  _inputs = []
  for i in (data, ):
    _add_input(i, _inputs)

  idx = omm.op_counter["Dropout"]
  omm.op_counter["Dropout"] += 1
  node = onnx.helper.make_node("Dropout",
                               _inputs, [f"_t_Dropout_{idx}"],
                               name=f"Dropout_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v10.AveragePool")
def AveragePool(X, **kwargs):
  _inputs = []
  for i in (X, ):
    _add_input(i, _inputs)

  idx = omm.op_counter["AveragePool"]
  omm.op_counter["AveragePool"] += 1
  node = onnx.helper.make_node("AveragePool",
                               _inputs, [f"_t_AveragePool_{idx}"],
                               name=f"AveragePool_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node