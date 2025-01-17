# Copyright (c) 2024 Advanced Micro Devices.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allpairs_reduce_scatter(gpuIds, numTbs, sizePerTb, sizePerRank, offset):
    ngpus = len(gpuIds)
    # sizePerTb = size // (numTbs * ngpus)
    # tbOffset = size // numTbs

    for r in range(ngpus):
        for tb in range(numTbs):
            # chunkIndex = offset + r * size + (sizePerTb * tb)
            # chunkIndex = offset + (tbOffset * tb) + (r * sizePerTb)
            # tb = index % numTbs
            # tb = index % ngpus
            # c = chunk(gpuIds[r], Buffer.input, chunkIndex)
            # c = chunk(gpuIds[r], Buffer.input, chunkIndex, sizePerTb)

            for peer in range(ngpus):
                if r != peer:
                    # peerChunkIndex = offset + peer * size + (sizePerTb * tb)
                    peerChunkIndex = offset + (tb * sizePerTb) + (peer * sizePerRank)
                    c = chunk(gpuIds[r], Buffer.input, peerChunkIndex, sizePerRank)
                    c.signal(gpuIds[peer], Buffer.input, peerChunkIndex, sendtb=tb)

    for r in range(ngpus):
        for tb in range(numTbs):
            # chunkIndex = offset + r * size + (sizePerTb * tb)
            chunkIndex = offset + (tb * sizePerTb) + (r * sizePerRank)
            c = chunk(gpuIds[r], Buffer.input, chunkIndex, sizePerRank)

            for peer in range(ngpus):
                if peer != r:
                    c.wait(gpuIds[peer], Buffer.input, chunkIndex, recvtb=tb)

            for peer in range(ngpus):
                if peer != r:
                    c.reduce(chunk(gpuIds[peer], Buffer.input, chunkIndex, sizePerRank), recvtb=tb)
 

def allpairs_all_gather(gpuIds, numTbs, sizePerTb, sizePerRank, offset):
    ngpus = len(gpuIds)
    # sizePerTb = size // numTbs

    for r in range(ngpus):
        for tb in range(numTbs):
            # tb = index % numTbs
            # tb = index % ngpus
            chunkIndex = offset + (tb * sizePerTb) + (r * sizePerRank)
            c = chunk(gpuIds[r], Buffer.input, chunkIndex, sizePerRank)
            for peer in range(ngpus):
                # peerIdx = peer if peer < r else (peer - 1)

                if peer != r:
                    # index = offset + r * size
                    c.put(gpuIds[peer], Buffer.input,  chunkIndex, sendtb=tb)
            
            for peer in range(ngpus):
                if peer != r:
                    c.signal(gpuIds[peer], Buffer.input, chunkIndex, sendtb=tb)

    # Each rank get final result from scratch space
    # for r in range(ngpus):
    #     for index in range(size):
            for peer in range(ngpus):
                # peerIdx = peer if peer < r else (peer - 1)

                peerChunkIndex = offset + (tb* sizePerTb) + (peer * sizePerRank)
                c = chunk(gpuIds[r], Buffer.input, peerChunkIndex, sizePerRank)
                if peer != r:
                    c.wait(gpuIds[peer], Buffer.input, peerChunkIndex, recvtb=tb)

# Performs two levels of allReduce
def allreduce(gpus, numTbs, instances):
    # nrows = gpusPerRank
    # ncols = gpus // nrows
    # gpusAcrossRank = gpus // gpusPerRank
    # ncols = gpusPerRank
    # nrows = gpus // ncols
    chunkperloop = gpus * numTbs #gpus * gpus
    topology = fully_connected(gpus)
    collective = AllReduce(gpus, chunkperloop, True)

    # innerSize = chunkperloop # // gpusPerRank
    # outerSize = chunkperloop // (gpusPerRank) # * gpusAcrossRank)

    size = chunkperloop

    sizePerTb = size // numTbs
    sizePerRank = sizePerTb // gpus

    with MSCCLPPProgram("hierarchical_allreduce",

        topology,
        collective,
        instances,
        protocol='Simple',
        # use_double_scratch_buffer=True,
    ):     
        
        # offset = 0

        gpuIds = []
        for n in range(gpus):
            gpuIds.append(n)
            # for m in range(nrows): # collect all GPU Ids in a column
            #     gpuIds.append( n * nrows + m)
            
        allpairs_reduce_scatter(gpuIds, numTbs, sizePerTb, sizePerRank, 0)

        allpairs_all_gather(gpuIds, numTbs, sizePerTb, sizePerRank, 0)

        Json()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
# parser.add_argument('num_gpus_per_rank', type=int, help ='number of gpus per rank')
parser.add_argument('num_tbs', type=int, help='number of threadblocks')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()

allreduce(args.num_gpus, args.num_tbs, args.instances)