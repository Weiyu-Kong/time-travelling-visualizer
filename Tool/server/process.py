import sys
import time
import numpy as np
import argparse

from utils import initialize_backend

sys.path.append('..')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process hyperparameters...')
    parser.add_argument('--content_path', type=str, default = '/home/kwy/project/time-travelling-visualizer-clean/training_dynamic/case_study')
    parser.add_argument('--vis_mothod', type=str, default = 'Trustvis')
    parser.add_argument('--setting', type=str, default = 'normal')

    args = parser.parse_args()

    print("Start getting training dynamic:")
    print("1. Preprocess")
    print("2. Train")
    print("3. Visualize")
    print("4. Evaluate")

    start = time.time()

    context, error_message_context = initialize_backend(args.content_path, args.vis_mothod, args.setting)
    context.strategy.visualize_embedding()

    end = time.time()
    print(f"Finish getting training dynamic! Duration: {end-start}")


