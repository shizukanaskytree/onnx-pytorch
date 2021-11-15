import argparse
import logging
import os
import re
import shutil
from collections import Counter
from collections import defaultdict

import numpy as np
import onnx
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import to_array
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from onnx_pytorch.code_gen_template import CodeGenTemplate
from onnx_pytorch.op_code_generators import *
from onnx_pytorch.utils.embedding_config_helper import load_embedding_config

print('onnx version: ', onnx.__version__)

class RenameHelper:

  def __init__(self, simplify_names=False):
    self.simplify_names = simplify_names

    self.tensor_name_mapping = {}
    self.tensor_name_counter = Counter()
    self.node_name_mapping = {}
    self.node_name_counter = Counter()

    self.tensor_counter = 0
    self.node_counter = Counter()

    self.init_name_set = set()
    self.sim_tensor_name_set = set()

  def get_tensor_name(self, tensor_name):
    if self.simplify_names:
      return self.get_simplify_tensor_name(tensor_name)
    if tensor_name.isnumeric():
      self.tensor_name_mapping[tensor_name] = f"t_{tensor_name}"
      return f"t_{tensor_name}"
    return tensor_name

  def get_node_name(self, node_name, op_type):
    if self.simplify_names or not node_name:
      return self.get_simplify_node_name(node_name, op_type)
    return f"n_{node_name}"

  def get_simplify_node_name(self, node_name, op_type):
    idx = self.node_counter[op_type]
    self.node_counter[op_type] += 1
    self.node_name_mapping[node_name] = f"n_{op_type}_{idx}"
    return self.node_name_mapping[node_name]

  def get_simplify_tensor_name(self, tensor_name):
    if tensor_name in self.tensor_name_mapping:
      return self.tensor_name_mapping[tensor_name]
    suffix = self.tensor_counter
    self.tensor_counter += 1
    sim_tensor_name = f"t_{suffix}"
    self.sim_tensor_name_set.add(sim_tensor_name)
    self.tensor_name_mapping[tensor_name] = sim_tensor_name
    return self.tensor_name_mapping[tensor_name]


class ModelCodeGenerator:

  def __init__(self,
               onnx_model=None,
               output_dir=None,
               simplify_names=False,
               tensor_inplace=False,
               continue_on_error=False,
               embedding_conf=None,
               shape_infer=True):
    self.onnx_model = onnx_model
    self.output_dir = output_dir
    self.rename_helper = RenameHelper(simplify_names)
    self.tensor_inplace = tensor_inplace
    self.continue_on_error = continue_on_error
    self.embedding_conf = embedding_conf
    self.shape_infer = shape_infer
    self.init_parts = []
    self.forward_parts = []
    self.method_parts = {}

    # dictionary containing adjacency List
    # to construct adjacency list by node's input and output
    # bfs topo traversing
    self.graph_adj_list = defaultdict(list)

    self.graph_input_val_info = {}
    self.graph_output_val_info = {}

    # all node name from onnx model
    self.model_nodes_name = []

    # num of nodes in the graph
    self.num_nodes = 0



  def cutting_node(self):
    max_pool = self.onnx_model.graph.node[4]
    print(max_pool)

  def fill_model_nodes_name(self):
    for node in self.onnx_model.graph.node:
      self.model_nodes_name.append(node.name)

    # print(self.model_nodes_name)

  def node_to_location(self):
    nodes_location = defaultdict(list)
    cnt = 0
    limit = 2
    loc = 0

    for node in self.onnx_model.graph.node:
      nodes_location[loc].append(node)
      if cnt == limit:
        loc += 1
      cnt += 1

    # print(nodes_location)
    return nodes_location



  # Propose:
  # 1. Allocate nodes to two groups, name is
  def partition(self, ):
    nodes_location = self.node_to_location()
    # specification: string: graph_def_proto
    partitions = {} # a map like GraphDef* dst_graph = &(*partitions)[dstp];
    num_partitions = len(nodes_location)

    # for i in range(num_partitions):
      # graph_def = onnx.GraphProto()

    # save the original initializer so I can reuse it without assigning one by one.
    node_name_init = {}
    for init in self.onnx_model.graph.initializer:
      node_name_init[init.name] = init


    # 试试小的图
    # 一切都以 name 为 id, 一切围绕 node 来做.
    graph_def = onnx.GraphProto()

    # 1, repeated NodeProto node = 1;
    for n in nodes_location[0]:
      print(n.name)
      graph_def.node.append(n)

    # 2, string name = 2;
    graph_def.name = 'p0_graph'

    # 5, repeated TensorProto initializer = 5;
    # 如何找出 node 的 initializer?
    # 思路: 把子图的 input 全部罗列出来, 然后 initializer 与子集 node inputs 的交集就是 子图的 initializer
    all_input_name = []
    all_output_name = []
    nodes_name = []
    for node in nodes_location[0]:
      for input_name in node.input:
        all_input_name.append(input_name)
      for output_name in node.output:
        all_output_name.append(output_name)
      nodes_name.append(node.name)

    inits = []
    for input_init in all_input_name:
      if input_init in node_name_init.keys():
        inits.append(input_init)
    # gist: https://www.notion.so/xiaofengwu/gist-node-of-all-their-input-all-initializer-initializer-f4ed549de00346499571a6b77b215be6

    # 往子图里面加入 initializer
    for init_name in inits:
      initializer = node_name_init[init_name]
      graph_def.initializer.append(initializer)

    # 15, repeated SparseTensorProto sparse_initializer = 15;
    # 这个为空

    # 10, string doc_string = 10;
    # 这个为空

    # 11,
    # The inputs and outputs of the graph.
    # repeated ValueInfoProto input = 11;
    # 思路:
    # ****************************
    # 有什么构造不出来的呢? 子集构造啊!
    # ****************************
    input_val_infos = []
    for i in self.onnx_model.graph.input:
      input_val_infos.append(i.name)




    print('end')






  def save_input_output_val_info(self):
    # repeated ValueInfoProto input = 11;
    for input_val_info in self.onnx_model.graph.input:
      print(input_val_info.name)
      self.graph_input_val_info[input_val_info.name] = input_val_info

    # print(self.graph_input_val_info)
    print('='*60)

    for output_val_info in self.onnx_model.graph.output:
      print(output_val_info.name)
      self.graph_output_val_info[output_val_info.name] = output_val_info

    # print(self.graph_output_val_info)
    """
    # repeated ValueInfoProto input = 11;

    data

    resnetv22_batchnorm0_gamma
    resnetv22_batchnorm0_beta
    resnetv22_batchnorm0_running_mean
    resnetv22_batchnorm0_running_var

    resnetv22_conv0_weight

    resnetv22_batchnorm1_gamma
    resnetv22_batchnorm1_beta
    resnetv22_batchnorm1_running_mean
    resnetv22_batchnorm1_running_var

    resnetv22_stage1_batchnorm0_gamma
    resnetv22_stage1_batchnorm0_beta
    resnetv22_stage1_batchnorm0_running_mean
    resnetv22_stage1_batchnorm0_running_var

    resnetv22_stage1_conv0_weight

    resnetv22_stage1_batchnorm1_gamma
    resnetv22_stage1_batchnorm1_beta
    resnetv22_stage1_batchnorm1_running_mean
    resnetv22_stage1_batchnorm1_running_var

    resnetv22_stage1_conv1_weight

    resnetv22_stage1_batchnorm2_gamma
    resnetv22_stage1_batchnorm2_beta
    resnetv22_stage1_batchnorm2_running_mean
    resnetv22_stage1_batchnorm2_running_var

    resnetv22_stage1_conv2_weight

    resnetv22_stage1_batchnorm3_gamma
    resnetv22_stage1_batchnorm3_beta
    resnetv22_stage1_batchnorm3_running_mean
    resnetv22_stage1_batchnorm3_running_var

    resnetv22_stage1_conv3_weight

    resnetv22_stage2_batchnorm0_gamma
    resnetv22_stage2_batchnorm0_beta
    resnetv22_stage2_batchnorm0_running_mean
    resnetv22_stage2_batchnorm0_running_var

    resnetv22_stage2_conv0_weight
    resnetv22_stage2_batchnorm1_gamma
    resnetv22_stage2_batchnorm1_beta
    resnetv22_stage2_batchnorm1_running_mean
    resnetv22_stage2_batchnorm1_running_var
    resnetv22_stage2_conv1_weight
    resnetv22_stage2_conv2_weight
    resnetv22_stage2_batchnorm2_gamma
    resnetv22_stage2_batchnorm2_beta
    resnetv22_stage2_batchnorm2_running_mean
    resnetv22_stage2_batchnorm2_running_var
    resnetv22_stage2_conv3_weight
    resnetv22_stage2_batchnorm3_gamma
    resnetv22_stage2_batchnorm3_beta
    resnetv22_stage2_batchnorm3_running_mean
    resnetv22_stage2_batchnorm3_running_var
    resnetv22_stage2_conv4_weight
    resnetv22_stage3_batchnorm0_gamma
    resnetv22_stage3_batchnorm0_beta
    resnetv22_stage3_batchnorm0_running_mean
    resnetv22_stage3_batchnorm0_running_var
    resnetv22_stage3_conv0_weight
    resnetv22_stage3_batchnorm1_gamma
    resnetv22_stage3_batchnorm1_beta
    resnetv22_stage3_batchnorm1_running_mean
    resnetv22_stage3_batchnorm1_running_var
    resnetv22_stage3_conv1_weight
    resnetv22_stage3_conv2_weight
    resnetv22_stage3_batchnorm2_gamma
    resnetv22_stage3_batchnorm2_beta
    resnetv22_stage3_batchnorm2_running_mean
    resnetv22_stage3_batchnorm2_running_var
    resnetv22_stage3_conv3_weight
    resnetv22_stage3_batchnorm3_gamma
    resnetv22_stage3_batchnorm3_beta
    resnetv22_stage3_batchnorm3_running_mean
    resnetv22_stage3_batchnorm3_running_var
    resnetv22_stage3_conv4_weight
    resnetv22_stage4_batchnorm0_gamma
    resnetv22_stage4_batchnorm0_beta
    resnetv22_stage4_batchnorm0_running_mean
    resnetv22_stage4_batchnorm0_running_var
    resnetv22_stage4_conv0_weight
    resnetv22_stage4_batchnorm1_gamma
    resnetv22_stage4_batchnorm1_beta
    resnetv22_stage4_batchnorm1_running_mean
    resnetv22_stage4_batchnorm1_running_var
    resnetv22_stage4_conv1_weight
    resnetv22_stage4_conv2_weight
    resnetv22_stage4_batchnorm2_gamma
    resnetv22_stage4_batchnorm2_beta
    resnetv22_stage4_batchnorm2_running_mean
    resnetv22_stage4_batchnorm2_running_var
    resnetv22_stage4_conv3_weight
    resnetv22_stage4_batchnorm3_gamma
    resnetv22_stage4_batchnorm3_beta
    resnetv22_stage4_batchnorm3_running_mean
    resnetv22_stage4_batchnorm3_running_var
    resnetv22_stage4_conv4_weight
    resnetv22_batchnorm2_gamma
    resnetv22_batchnorm2_beta
    resnetv22_batchnorm2_running_mean
    resnetv22_batchnorm2_running_var

    reshape_attr_tensor164

    resnetv22_dense0_weight
    resnetv22_dense0_bias
    ============================================================

    # repeated ValueInfoProto output = 12;
    resnetv22_dense0_fwd
    """

  def group_0_nodes(self):
    """
    这个部分可以用 traversing 的方式去分开 nodes, 再试试.
    """
    p0_nodes = []
    for n in self.onnx_model.graph.node[:2]:
      p0_nodes.append(n)
    return p0_nodes

  def group_1_nodes(self):
    p1_nodes = []
    for n in self.onnx_model.graph.node[2:]:
      p1_nodes.append(n)
    return p1_nodes

  def who_are_inputs(self):
    """
    Nodes without input are inputs.
    Output dst are not in the node set within itself are 断边
    Input src node are not in the node set itself are 断边

    难点:
    The inputs and outputs of the graph.
    repeated ValueInfoProto input = 11;
    repeated ValueInfoProto output = 12;
    """


    ...


  def construct_adj_list_partitioned(self,):
    """ construct adj list after partitioning
    """
    # 目前是人为区分了两组 nodes 分组

    # self.preprocess_onnx_model()
    for n in self.onnx_model.graph.node[:2]:
      for in_node in n.input:
        self.add_edge(in_node, n.name)

    print('adj list', self.graph_adj_list)

  # Topological Sorting
	# function to add an edge to graph adjacency list
  def add_edge(self, src, dst):
	  self.graph_adj_list[src].append(dst)

  def construct_adj_list(self):
    # self.preprocess_onnx_model()

    # 消灭 initializer 这样的边! 如何搞?
    for n in self.onnx_model.graph.node:
      # in edge (in node) 是对的, out edge 是错的, gist: https://www.notion.so/xiaofengwu/gist-node-input-output-names-23860113b0aa47068a7b46b213ea91b2
      for in_node in n.input:
        if in_node in self.model_nodes_name:
          self.add_edge(in_node, n.name)

    print('adj list', self.graph_adj_list)


  # traversing, to study
  # topo sort
  def traverse_graph(self):
    ...



  def add_init_part(self, m):
    if type(m) in (list, tuple, set):
      self.init_parts.extend(m)
    else:
      self.init_parts.append(m)

  def add_forward_part(self, m):
    if type(m) in (list, tuple, set):
      self.forward_parts.extend(m)
    else:
      self.forward_parts.append(m)

  def add_forward_return(self, outputs_value_infos):
    return_list = [
        f"{self.rename_helper.get_tensor_name(o.name)}"
        for o in outputs_value_infos
    ]
    self.forward_parts.append(f"return {', '.join(return_list)}")

  def add_forward_input(self, inputs_value_infos):
    initializer_names = {i.name for i in self.onnx_model.graph.initializer}
    return_list = [
        f"{self.rename_helper.get_tensor_name(i.name)}"
        for i in inputs_value_infos
        if i.name not in initializer_names
    ]
    if len(return_list) == 1:
      self.forward_parts.append(f"{return_list[0]}, = inputs")
    else:
      self.forward_parts.append(f"{', '.join(return_list)} = inputs")

  def gen_model_code(self):
    return CodeGenTemplate.model(model_init='''
    '''.join(self.init_parts),
                                 model_forward='''
    '''.join(self.forward_parts),
                                 model_method='''
  '''.join(self.method_parts.values()),
                                 test_run_model=self.gen_test_run_model_code())

  def gen_test_run_model_code(self):
    numpy_input_str = []
    initializer_names = {i.name for i in self.onnx_model.graph.initializer}
    for i in self.onnx_model.graph.input:
      if i.name in initializer_names:
        continue
      dtype = TENSOR_TYPE_TO_NP_TYPE[i.type.tensor_type.elem_type]
      shape = []
      for d in i.type.tensor_type.shape.dim:
        if d.dim_param != "":
          shape.append(1)
        else:
          shape.append(d.dim_value)
      if shape:
        numpy_input_str.append(
            f"torch.from_numpy(np.random.randn(*{[s if s > 1 else 1 for s in shape].__repr__()}).astype(np.{dtype.name}))"
        )
      else:
        numpy_input_str.append(
            f"torch.from_numpy(np.random.randn(1).astype(np.{dtype.name}))")
    test_run_model = [
        f'''@torch.no_grad()
def test_run_model(inputs=[{', '.join(numpy_input_str)}]):''',
        "model = Model()", "model.eval()"
    ]
    test_run_model.extend(["rs = model(*inputs)", "print(rs)", "return rs"])
    return '''
  '''.join(test_run_model)

  def preprocess_onnx_model(self):
    for n in self.onnx_model.graph.node:
      inputs, outputs = [], []
      for ls, f in ((inputs, n.input), (outputs, n.output)):
        for i in f:
          new_i = re.sub("[:/.]", "_", i)
          ls.append(new_i)
          if i != ls[-1] and not self.rename_helper.simplify_names:
            logging.info(f"Tensor name {i} is changed to {ls[-1]}.")
          self.rename_helper.tensor_name_counter[ls[-1]] += 1

      n.ClearField("input")
      n.input.extend(inputs)
      n.ClearField("output")
      n.output.extend(outputs)

      old_name = n.name
      n.name = re.sub("[:/.]", "_", n.name)
      if old_name != n.name and not self.rename_helper.simplify_names:
        logging.info(f"Node name {old_name} is changed to {n.name}.")
      self.rename_helper.node_name_counter[n.name] += 1

    for f in (self.onnx_model.graph.input, self.onnx_model.graph.output,
              self.onnx_model.graph.initializer):
      for i in f:
        old_name = i.name
        i.name = re.sub("[:/.]", "_", i.name)
        if old_name != i.name and not self.rename_helper.simplify_names:
          logging.info(f"Tensor name {i.name} is changed to {i.name}.")
        self.rename_helper.tensor_name_counter[i.name] += 1

    model = self.onnx_model
    for f in (model.graph.input, model.graph.output):
      for i in f:
        for d in i.type.tensor_type.shape.dim:
          if d.dim_param != "":
            d.dim_param = ""
            d.dim_value = -1
          elif d.dim_value == 0:
            d.dim_value = -1
    # TODO how to deal with custom op?
    if self.shape_infer:
      try:
        model.graph.ClearField("value_info")
        model = SymbolicShapeInference.infer_shapes(model, 2**31 - 1, True,
                                                    True, 1)
      except:
        logging.warning("Shape infer by onnxruntime failed.")
    else:
      for f in (self.onnx_model.graph.value_info,):
        for i in f:
          old_name = i.name
          i.name = re.sub("[:/.]", "_", i.name)
          if old_name != i.name and not self.rename_helper.simplify_names:
            logging.info(f"Tensor name {i.name} is changed to {i.name}.")
          self.rename_helper.tensor_name_counter[i.name] += 1
    onnx.save(model, os.path.join(self.output_dir, "tmp_processed.onnx"))
    self.onnx_model = model

  def add_attr_to_op_code_generator(self, op_code_gen):
    for k, v in {
        "rename_helper": self.rename_helper,
        "tensor_inplace": self.tensor_inplace,
        "embedding_conf": self.embedding_conf
    }.items():
      if hasattr(op_code_gen, k):
        setattr(op_code_gen, k, v)


  def construct_partitions(self):
    self.preprocess_onnx_model()

    # 信息的源头
    model = self.onnx_model
    # content: https://gist.github.com/shizukanaskytree/beac660dfc9aa8be92e3308bade0b617
    # skeleton: https://github.com/onnx/onnx/blob/master/onnx/onnx-ml.proto3

    # model.ir_version
    # model.opset_import

    # The nodes in the graph, sorted topologically.
    # repeated NodeProto node = 1;
    nodes = []
    print(f'nodes num: {len(model.graph.node)}') # nodes num: 69
    for n in model.graph.node:
      nodes.append(n)

    partition_cnt = 2 # input data 不包括在内
    p0_nodes = []
    p0_nodes.extend(nodes[:partition_cnt])
    p1_nodes = []
    p1_nodes.extend(nodes[partition_cnt:])

    # todo:
    # nodes' <==> nodes'
    # How to find the edges between these two cluster?
    # self detection in its own cluster.
    # all edges in cluster 0
    # all edges in cluster 1

    # 都靠遍历
    # group 1's output edges set
    # group 2's input edges set
    # if the output edge is in another group's input edges set, they are connect.



    # nodes -> nodes' name, inputs' name
    # 奇怪的是,node 的 input 居然不仅仅是 input
    p0_nodes_inputs_name = []
    for n in p0_nodes:
      for i in n.input:
        p0_nodes_inputs_name.append(i)
    """
    p0_nodes_inputs_name
    ['data', 'resnetv22_batchnorm0_gamma', 'resnetv22_batchnorm0_beta', 'resnetv22_batchnorm0...nning_mean', 'resnetv22_batchnorm0...unning_var', 'resnetv22_batchnorm0_fwd', 'resnetv22_conv0_weight']
    special variables:
    function variables:
    0: 'data'
    1: 'resnetv22_batchnorm0_gamma'
    2: 'resnetv22_batchnorm0_beta'
    3: 'resnetv22_batchnorm0_running_mean'
    4: 'resnetv22_batchnorm0_running_var'
    5: 'resnetv22_batchnorm0_fwd'
    6: 'resnetv22_conv0_weight'
    len(): 7
    """

    # partition all inputs based on the partitioned nodes' input
    map_input_ValueInfoProto = {}
    for input_val_info in model.graph.input:
      map_input_ValueInfoProto[input_val_info.name] = input_val_info


    p0_input_ValueInfoProto = [] # element type is <class 'onnx.onnx_ml_pb2.ValueInfoProto'>
    for name in p0_nodes_inputs_name:
      # error.
      # assert name in map_input_ValueInfoProto.keys()
      if name in map_input_ValueInfoProto.keys():
        p0_input_ValueInfoProto.append(map_input_ValueInfoProto[name])


    # 0
    p0 = onnx.ModelProto()
    p0.ir_version = model.ir_version
    # http://www.yeolar.com/note/2014/12/23/protobuf-python-generated/
    p0.opset_import.extend(model.opset_import)
    p0.producer_name = model.producer_name
    p0.producer_version = model.producer_version
    p0.domain = model.domain
    p0.model_version = model.model_version
    p0.doc_string = model.doc_string
    p0.metadata_props.extend(model.metadata_props)
    p0.training_info.extend(model.training_info)
    p0.functions.extend(model.functions)

    # 这才是重头戏.
    p0_graph = onnx.GraphProto()
    p0_graph.extend(nodes[:partition_cnt])
    p0_graph.name = "p0_graph"
    # input
    p0_graph.input.extend(p0_input_ValueInfoProto)

    # =============================================================
    # output
    # =============================================================
    # output is the output of the last node

    # 我感觉我需要明确 edge being cut.
    # 如果明确了那个 edge 或者那些 edges, 就可以了.

    # 1
    p1 = onnx.ModelProto()



  def run(self):
    self.preprocess_onnx_model()

    initializers = {i.name: i for i in self.onnx_model.graph.initializer}

    input_value_infos = {i.name: i for i in self.onnx_model.graph.input}
    output_value_infos = {i.name: i for i in self.onnx_model.graph.output}

    value_infos = {}
    value_infos.update(input_value_infos)
    value_infos.update(output_value_infos)

    value_infos.update({i.name: i for i in self.onnx_model.graph.value_info})

    for i in self.onnx_model.graph.initializer:
      self.rename_helper.get_tensor_name(i.name)

    self.add_forward_input(self.onnx_model.graph.input)

    for n in self.onnx_model.graph.node:
      op_code_gen = get_op_code_generator(n.op_type)
      self.add_attr_to_op_code_generator(op_code_gen)

      if op_code_gen is None:
        if self.continue_on_error:
          self.add_forward_part(n.__repr__())
          logging.warning(f"OpCodeGenerator is unimplemented for {n.op_type}. "
                          "Please modify this part by manual later.")
        else:
          raise NotImplementedError(
              f"OpCodeGenerator is unimplemented for {n.op_type}.")
      else:
        try:
          if hasattr(op_code_gen,
                     "gen_method") and n.op_type not in self.method_parts:
            self.method_parts[n.op_type] = op_code_gen.gen_method()

          gened = op_code_gen.gen(n, value_infos, initializers)

          self.add_init_part(gened["init"])
          self.add_forward_part(gened["forward"])
        except BaseException as e:
          if self.continue_on_error:
            logging.warning(e)
            self.add_forward_part(n.__repr__())
          else:
            raise e

    self.add_forward_return(self.onnx_model.graph.output)

    gened_code = self.gen_model_code()

    print(gened_code)

    with open(os.path.join(self.output_dir, "model.py"), "w") as f:
      f.write(gened_code)

    shutil.rmtree(os.path.join(self.output_dir, "variables"),
                  ignore_errors=True)

    os.makedirs(os.path.join(self.output_dir, "variables"))

    for k, v in initializers.items():
      np.save(
          os.path.join(self.output_dir, "variables",
                       f"{self.rename_helper.get_tensor_name(k)}.npy"),
          to_array(v))


def gen(
    onnx_model,
    output_dir,
    overwrite=False,
    tensor_inplace=False,
    simplify_names=False,
    continue_on_error=False,
    embedding_conf_file=None,
    shape_infer=True,
):
  model_code_generator = get_model_code_generator(
      onnx_model, output_dir, overwrite, tensor_inplace, simplify_names,
      continue_on_error, embedding_conf_file, shape_infer)
  # original code
  # model_code_generator.run()


  # my testing ...
  # model_code_generator.construct_partitions()
  # model_code_generator.construct_adj_list()

  # model_code_generator.construct_adj_list_partitioned()
  # model_code_generator.save_input_output_val_info()
  # model_code_generator.node_to_location()
  # model_code_generator.partition()

  model_code_generator.preprocess_onnx_model()
  # model_code_generator.fill_model_nodes_name()
  model_code_generator.construct_adj_list()


  # model_code_generator.cutting_node()


def get_model_code_generator(
    onnx_model,
    output_dir,
    overwrite=False,
    tensor_inplace=False,
    simplify_names=False,
    continue_on_error=False,
    embedding_conf_file=None,
    shape_infer=False,
):
  kwargs = {
      "output_dir": output_dir,
      "simplify_names": simplify_names,
      "tensor_inplace": tensor_inplace,
      "continue_on_error": continue_on_error,
      "shape_infer": shape_infer
  }
  if type(onnx_model) == onnx.ModelProto:
    kwargs["onnx_model"] = onnx_model
  else:
    assert os.path.exists(
        onnx_model), f"ONNX model {onnx_model} does not exist."
    assert os.path.isfile(onnx_model), f"{onnx_model} is not a file."
    assert os.path.exists(
        output_dir
    ) and overwrite is not True, f"{output_dir} is not empty and overwrite is not True."
    assert os.path.isdir(output_dir), f"{output_dir} is not directory."
    kwargs["onnx_model"] = onnx.load(onnx_model)
  if overwrite:
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
  if embedding_conf_file is not None:
    assert os.path.exists(
        embedding_conf_file
    ), f"Embedding config file {embedding_conf_file} does not exist."
    kwargs["embedding_conf"] = load_embedding_config(embedding_conf_file)
  return ModelCodeGenerator(**kwargs)


def main():
  debug = True
  parser = argparse.ArgumentParser()
  parser.add_argument("--onnx_model_path",
                      default=None,
                      type=str,
                      required=not debug,
                      help="The ONNX model path.")
  parser.add_argument("--output_dir",
                      default=None,
                      type=str,
                      required=not debug,
                      help="The output dir")
  parser.add_argument("--overwrite",
                      default=False,
                      type=bool,
                      help="Should overwrite the output dir.")
  parser.add_argument("--tensor_inplace",
                      default=False,
                      type=bool,
                      help="Try best to inplace tensor.")
  parser.add_argument("--continue_on_error",
                      default=False,
                      type=bool,
                      help="Continue on error.")
  parser.add_argument("--embedding_conf_file",
                      type=str,
                      help="Embedding config file path.")
  parser.add_argument(
      "--simplify_names",
      default=False,
      type=int,
      help="Use indexing shorten name instead of original name.")
  args = parser.parse_args()

  # 入口
  gen(onnx_model=args.onnx_model_path,
      output_dir=args.output_dir,
      overwrite=args.overwrite,
      tensor_inplace=args.tensor_inplace,
      simplify_names=args.simplify_names,
      continue_on_error=args.continue_on_error,
      embedding_conf_file=args.embedding_conf_file)


if __name__ == '__main__':
  main()
