from new_models.GraphCL import *

# adding new methods
new_models = [GraphCL]

"""
python netBenchmark.py --method=graphcl --task_method=task1 --tuning_method=random

python netBenchmark.py --method=graphcl --task_method=task1 --training_ratio=0.1 --tuning_method=random

python netBenchmark.py --method=graphcl --task_method=task1 --training_ratio=0.1 --tuning_method=tpe

python netBenchmark.py --method=graphcl --task_method=task2 --training_ratio=0.1 --tuning_method=random

python netBenchmark.py --method=graphcl --task_method=task2 --training_ratio=0.1 --tuning_method=tpe

python netBenchmark.py --method=graphcl --task_method=task3 --training_ratio=0.1 --tuning_method=random

python netBenchmark.py --method=graphcl --task_method=task3 --training_ratio=0.1 --tuning_method=tpe
"""