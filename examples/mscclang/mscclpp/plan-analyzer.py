import sys
import json
import argparse
# from pprint import pprint

def analyze_json_file(file_name):
    with open(file_name) as f:
        data = json.load(f)
        ngpus = len(data["gpus"])

        print("#GPUs: ", ngpus)
        threadblocks = [0] * ngpus
        max_ops = [0] * ngpus
        min_ops = [sys.maxsize] * ngpus
        nchannels = [0] * ngpus
        max_tb_channels = [0] * ngpus
        max_tb_id = [0] * ngpus
        max_op_channels = [0] * ngpus
        invalid_nop = [False] * ngpus
        for g in range(ngpus):
            num_tb = len(data["gpus"][g]["threadblocks"])
            threadblocks[g] = num_tb
            nchannels[g] = len(data["gpus"][g]["channels"])

            for tb in range(num_tb):
                tb_id = data["gpus"][g]["threadblocks"][tb]["id"]
                if tb_id > max_tb_id[g]:
                    max_tb_id[g] = tb_id
                num_ops = len(data["gpus"][g]["threadblocks"][tb]["ops"])
                num_channels = len(data["gpus"][g]["threadblocks"][tb]["channels"])
                if num_ops > max_ops[g]:
                    max_ops[g] = num_ops
                if num_ops < min_ops[g]:
                    min_ops[g] = num_ops
                if num_channels > max_tb_channels[g]:
                    max_tb_channels[g] = num_channels
                for op in range(num_ops):
                    op_data = data["gpus"][g]["threadblocks"][tb]["ops"][op]
                    input_indexes = 0
                    output_indexes = 0

                    if 'i_cids' in op_data:
                        input_indexes = len(data["gpus"][g]["threadblocks"][tb]["ops"][op]['i_cids'])
                    if 'srcs' in op_data:
                        input_indexes = len(data["gpus"][g]["threadblocks"][tb]["ops"][op]['srcs'])

                    if 'o_cids' in op_data:
                        output_indexes = len(data["gpus"][g]["threadblocks"][tb]["ops"][op]['o_cids'])
                    if 'dsts' in op_data:
                        output_indexes = len(data["gpus"][g]["threadblocks"][tb]["ops"][op]['dsts'])
                    max_indexes = max(input_indexes, output_indexes)
                    if max_indexes > max_op_channels[g]:
                        max_op_channels[g] = max_indexes
                    
                    if op_data["name"] == "nop":
                        num_deps = len(op_data["deps"])
                        for d in range(num_deps):
                            if op_data["deps"][d]["tb"] != tb_id:
                                invalid_nop[g] = True



        print("GPU | #TBs | Min #OPs | Max #OPs | #Channels | Max TB chls | Max OP chls | Max TB ID | Invalid NOP |")
        for g in range(ngpus):
            if g < 10:
                print(" ", end='')
            print(g, " | ", threadblocks[g], " |     ", min_ops[g], " |     ", max_ops[g], " |       ", nchannels[g], " |         ", max_tb_channels[g], " |        ", max_op_channels[g], " |        ", max_tb_id[g], " |     ", invalid_nop[g])
            # print("--- GPU ", g, " ---")
            # print("#threadblocks, ", threadblocks[g])
            # print("Max operations, ", max_ops[g])

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help="json file to analyze")

args = parser.parse_args()
analyze_json_file(args.file)