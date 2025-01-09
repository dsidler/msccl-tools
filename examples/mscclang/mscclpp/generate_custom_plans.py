import os
import subprocess
import math

def createdir(base_path: str, dir_name : str) -> str:
    new_path = os.path.join(base_path, dir_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path

# variables
dir_name = 'custom-plans'
ngpus = [64]
algos = ['hierarchical', 'recursive-hd']
protocols = ['LL', 'Simple']
tbs = [1, 2, 4, 8]
instances = [1]
algoMap = {
    'LL' : {'hierarchical' : "allreduce_hierarchical_packet1e",
            'recursive-hd' : "allreduce_recursive-hd_packet1",},
    'Simple' : {'hierarchical' : "allreduce_hierarchical1e",
                'recursive-hd' : "allreduce_recursive-hd1",},
}


# Check if custom plans directory exists
# Create if not
current_dir = os.getcwd()
base_path = os.path.join(current_dir, dir_name)
if os.path.exists(base_path):
    exit
os.makedirs(base_path)


for algo in algos:
    for protocol in protocols:
        if protocol not in algoMap:
            print("No python file defined for protocol ", protocol)
            continue
        if algo not in algoMap[protocol]:
            print("No python file defined for protocol ", protocol, " and algo ", algo)
            continue

        # Create directory
        algo_path = createdir(base_path, algo)
        protocol_path = createdir(algo_path, protocol)

        for tb in tbs:
            tb_path = createdir(protocol_path, str(tb))
            for inst in instances:
                script_file = os.path.join(current_dir, algoMap[protocol][algo])
                args = ['python']
                args.append(script_file + '.py')
                gpus = ngpus[0]
                gpusPerRank = str(int(math.sqrt(gpus)))

                print("Generate custom plan for ", algo, " on ", gpus, " GPUs using ", protocol, " with ", tb, " threadblocks and ", inst, " instances.")

                sub_args = ['allreduce']
                sub_args.append(str(gpus))
                if algo == 'hierarchical':
                    sub_args.append(gpusPerRank)
                sub_args.append(str(tb))
                sub_args.append(str(inst))

                json_file = '_'.join(sub_args)
                json_file += '.json'
                json_path = os.path.join(tb_path, json_file)

                args.extend(sub_args[1:])
                # args.append('> ' + json_path)
                print("ARGS: ", args)
                with open(json_path, 'w') as output_file:
                    subprocess.call(args, stdout=output_file)


