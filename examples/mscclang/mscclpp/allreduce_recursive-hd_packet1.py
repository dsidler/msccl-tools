# Copyright (c) 2024 Advanced Micro Devices.
# Licensed under the MIT License.

import math
import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def recursive_halving_reduce_scatter(gpuIds, numTbs, sizePerTb, sizerPerRank, offset):
    ngpus = len(gpuIds)

    for r1 in range(ngpus):
        for tb in range(numTbs):
            for r2 in range(ngpus):
                if r1 != r2:
                    chunkIndex = offset + (tb * sizePerTb) + (r2 * sizerPerRank)
                    scratchChunkIndex = offset + (tb * sizePerTb) + (r1 * sizerPerRank)
                    c = chunk(gpuIds[r1], Buffer.input, chunkIndex, sizerPerRank)
                    c.put_packet(gpuIds[r2], "scratch", scratchChunkIndex, sendtb=tb)

    for r in range(ngpus):
        for tb in range(numTbs):
            chunkIndex = offset + (tb * sizePerTb) + (r * sizerPerRank)
            c = chunk(gpuIds[r], Buffer.input, chunkIndex, sizerPerRank)
            for peer in range(ngpus):
                scratchChunkIndex = offset + (tb * sizePerTb) + (peer * sizerPerRank)
                if peer != r:
                    c.reduce_packet(chunk(gpuIds[r], "scratch", scratchChunkIndex, sizerPerRank), recvtb=tb)
  


def recursive_doubling_all_gather(gpuIds, numTbs, sizePerTb, sizePerRank, offset):
    ngpus = len(gpuIds)

    for r in range(ngpus):
        for tb in range(numTbs):
            index = offset + (tb * sizePerTb) + (r * sizePerRank)
            c = chunk(gpuIds[r], Buffer.input, index, sizePerRank)
            for peer in range(ngpus):
                if peer != r:
                    c.put_packet(gpuIds[peer], "scratch", index, sendtb=tb)

    for r in range(ngpus):
        for tb in range(numTbs):
            for peer in range(ngpus):
                if peer != r:
                    index = offset + (tb * sizePerTb) + (peer * sizePerRank)
                    c = chunk(gpuIds[r], "scratch", index, sizePerRank)
                    c.copy_packet(gpuIds[r], Buffer.input, index, sendtb=tb)


# Performs Recursive-halving/doubling allReduce
# Assumes gpus is power of 2
def recursive_hd_allreduce(gpus, numTbs, instances):
    pairsize = 2
    npairs = gpus // pairsize
    chunkperloop = gpus * numTbs # * gpus
    topology = fully_connected(gpus)
    collective = AllReduce(gpus, chunkperloop, True)
    depth = int(math.log2(gpus))
    if 2**depth != gpus:
        raise ValueError('Number of GPUs is not power of 2')

    with MSCCLPPProgram("recursive-hd_allreduce",
        topology,
        collective,
        instances,
        protocol='LL',
        use_double_scratch_buffer=True,
    ):     
        
        # Reduce-Scatter on each column first, assumption being GPUs in a column have faster connectivity - NVLINK
        # Each GPU exchanges (nrows - 1) * 1/rows of data with other GPUs in the same column 
        # After this step, first GPU in each column will have 1st 1/nrows, 2nd GPU will have 2nd of 1/nrows data reduced
        # size = chunkperloop // pairsize
        sizePerTb = chunkperloop // numTbs
        sizePerRank = sizePerTb // pairsize
        offsets = [0] * npairs
        npairsPerSplit = npairs
        for level in range(depth):
            stride = gpus // (2**level)
            distance = gpus // (2 ** (level + 1))
            for splits in range(2**level):
                for n in range(npairsPerSplit):
                    gpuIds = []
                    for m in range(pairsize): # collect all GPU Ids in a column
                        gpuIds.append( (n + (splits * stride)) + m * distance)
                    
                    recursive_halving_reduce_scatter(gpuIds, numTbs, sizePerTb, sizePerRank, offsets[splits*npairsPerSplit + n])

                    # The upper half of pairs will need an updated offset in next round
                    if n >=  (npairsPerSplit // 2):
                        offsets[splits*npairsPerSplit + n] += sizePerRank

            npairsPerSplit //= 2
            # size //= 2
            # sizePerTb //= 2
            sizePerRank //= 2

        # AllGather: AllGather phase goes in reverse order, first gather across rows of GPU
        # After this step, Each GPU in a rows have 1/ncols of data
        # size = numTbs #TODO size * 2
        sizePerRank = 1
        npairsPerSplit = 1
        for rlevel in range(depth):
            level = depth - rlevel - 1
            stride = gpus // (2**level)
            distance = gpus // (2 ** (level + 1))
            for splits in range(2**level):
                for n in range(npairsPerSplit):
                    # The upper half of pairs will need an updated offset in next round
                    if n >=  (npairsPerSplit // 2):
                        offsets[splits*npairsPerSplit + n] -= sizePerRank

                    gpuIds = []
                    for m in range(pairsize): # collect all GPU Ids in a column
                        gpuIds.append( (n + (splits * stride)) + m * distance)
                    
                    recursive_doubling_all_gather(gpuIds, numTbs, sizePerTb, sizePerRank, offsets[splits*npairsPerSplit + n])


            npairsPerSplit *= 2
            # size *= 2
            sizePerRank *= 2

        Json()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('num_tbs', type=int, help='number of threadblocks')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()

recursive_hd_allreduce(args.num_gpus, args.num_tbs, args.instances)