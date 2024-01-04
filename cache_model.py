import PerformancePipeline
import argparse
from IPython.display import clear_output
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='SG161222/Realistic_Vision_V2.0', help='model name')
args = parser.parse_args()
model = PerformancePipeline.from_pretrained(args.model_name)
del model
clear_output()
print("Cached model %s." % args.model_name)