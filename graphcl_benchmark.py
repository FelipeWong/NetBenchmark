from netBenchmark import *
from model_graphcl import *

modellist.append(GraphCL)
# print(modellist)
modeldict["GraphCL"] = GraphCL
modeldict_all["GraphCL"] = GraphCL
# print(modeldict)
# print(modeldict_all)


# python graphcl_benchmark.py --method=GraphCL --dataset=cora --task_method=task1 --cuda_device=1

try:
    main(parse_args())
except Exception as e:
    print("An error occurred:", e)
