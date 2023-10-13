from typing import List, Tuple
from models.resnet import ResNet50
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps, Device, Compiled
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.search import bufs_from_lin, time_linearizer, get_linearizer_actions
from tinygrad.helpers import ansilen, DEBUG, getenv
from tinygrad.graph import print_tree
from tinygrad.lazy import vars_from_ast
from tinygrad.shape.symbolic import sym_infer

import shelve
global_db = shelve.open("/tmp/greedy_cache")

if __name__ == "__main__":
  mdl = ResNet50()
  seen = set()

  # the device we are optimizing for
  device: Compiled = Device[Device.DEFAULT]
  print(f"optimizing for {Device.DEFAULT}")

  # first model run to init the weights, they are saved in seen
  mdl(Tensor.empty(64, 3, 224, 224)).lazydata.schedule(seen)

  # run model again to get only what changes, these are the kernels of the model
  x = Tensor.empty(64, 3, 224, 224)
  out = mdl(x)
  sched = out.lazydata.schedule(seen)
  sched = [x for x in sched if x.ast.op not in LoadOps]

  # work with the schedule
  total_tm = 0
  running_gflops = 0
  for i,si in enumerate(sched[:20]): # 110 ms
    if DEBUG >= 2: print_tree(si.ast)

    # create output/input buffers (NOTE: bufs_from_lin is slower, so we don't use it. TODO: fix)
    rawbufs = [device.buffer(si.out.st.size(), si.out.dtype)] + [device.buffer(x.st.size(), x.dtype) for x in si.inputs]
    #rawbufs = bufs_from_lin(lin)

    # "linearize" the op into uops in different ways
    lins:List[Linearizer] = []

    # always try hand coded opt
    lin = Linearizer(si.ast, device.linearizer_opts)
    lin.hand_coded_optimizations()
    lins.append(lin)

    # maybe try tensor cores
    lin = Linearizer(si.ast, device.linearizer_opts)
    if lin.apply_tensor_cores():
      lins.append(lin)

    # try a greedy search
    if getenv("GREEDY"):
      lin = Linearizer(si.ast, device.linearizer_opts)
      if str(lin.ast) in global_db:
        for ao in global_db[str(lin.ast)]:
          lin.apply_opt(ao)
        lins.append(lin)
      else:
        BEAM = 4

        beams: List[Tuple[Linearizer, List[int], float]] = [(lin, [], float('inf'))] # lin, opts, time
        while 1:
          experiments: List[Tuple[Linearizer, List[int], float]] = []
          for l, opts, _ in beams:
            acted_lins = get_linearizer_actions(l)
            for k, v in acted_lins.items():
              experiments.append((v, opts + [k], time_linearizer(v, rawbufs)))
          experiments = sorted(experiments, key=lambda x: x[2])
          if sum(t for _, _, t in experiments[:BEAM]) * 1.01 > sum(t for _, _, t in beams): lins.append(beams[0][0]); break
          beams = experiments[:BEAM]

          if DEBUG >= 1:
            for l, opts, t in beams: print(f"{t*1e3:10.2f} ms from {len(opts):3d} actions", l.colored_shape())

    # benchmark the programs
    choices = []
    for lin in lins:
      tm = time_linearizer(lin, rawbufs, allow_test_size=False, cnt=10, should_copy=False)
      gflops = sym_infer(lin.info.flops, {k:k.min for k in vars_from_ast(lin.ast)})*1e-9/tm
      choices.append((tm, gflops, lin))

      # print all kernels
      if DEBUG >= 1: print(f"                 kernel {i:2d} {lin.display_name+' '*(37-ansilen(lin.display_name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")
    tm, gflops, lin = sorted(choices, key=lambda x: x[0])[0]
    print(f"*** {total_tm*1000:7.2f} ms : kernel {i:2d} {lin.display_name+' '*(37-ansilen(lin.display_name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")
    total_tm += tm
    running_gflops += gflops * tm
  print(f"******* total {total_tm*1000:.2f} ms, {running_gflops/total_tm:6.0f} GFLOPS")
