# Copyright (c) 2024 Advanced Micro Devices.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allpairs_reduce_scatter(gpuIds, numTbs, sizePerTb, sizePerRank, offset):
    ngpus = len(gpuIds)

    # Each rank sends the nth chunk to the nth rank into scratch space
    for r in range(ngpus):
        for tb in range(numTbs):
            for peer in range(ngpus):
                if r != peer:
                    local_rank = gpuIds[r]
                    remote_rank = gpuIds[peer]
                    chunkIndex = offset + (tb * sizePerTb) + (r * sizePerRank)
                    peerChunkIndex = offset + (tb * sizePerTb) + (peer * sizePerRank)
                    c = chunk(local_rank, Buffer.input, peerChunkIndex, sizePerRank)
                    c.put_packet(remote_rank, "scratch", index=chunkIndex, sendtb=tb)

    for r in range(ngpus):
        for tb in range(numTbs):
            chunkIndex = offset + (tb * sizePerTb) + (r * sizePerRank)

            c = chunk(gpuIds[r], Buffer.input, chunkIndex, sizePerRank)
            for peer in range(ngpus):
                peerChunkIndex = offset + (tb * sizePerTb) + (peer * sizePerRank)
                if peer != r:
                    c.reduce_packet(chunk(gpuIds[r], "scratch", peerChunkIndex, sizePerRank), recvtb=tb)


def allpairs_all_gather(gpuIds, numTbs, sizePerTb, sizePerRank, offset, dataSize):
    ngpus = len(gpuIds)

    for r in range(ngpus):
        for tb in range(numTbs):
            chunkIndex = offset + (tb * sizePerTb) + (r * sizePerRank)
            c = chunk(gpuIds[r], Buffer.input, chunkIndex, sizePerRank)
            for peer in range(ngpus):
                if peer != r:
                    c.put_packet(gpuIds[peer], "scratch", (dataSize * dataSize) + chunkIndex, sendtb=tb)

    # Each rank get final result from scratch space
    for r in range(ngpus):
        for tb in range(numTbs):
            for peer in range(ngpus):
                if peer != r:
                    peerChunkIndex = offset + (tb * sizePerTb) + (peer * sizePerRank)
                    c = chunk(gpuIds[r], "scratch", (dataSize * dataSize) + peerChunkIndex, sizePerRank)
                    c.copy_packet(gpuIds[r], Buffer.input, peerChunkIndex, sendtb=tb)

# Performs two levels of allReduce
def allreduce(gpus, numTbs, instances):
    # nrows = gpusPerRank
    # ncols = gpus // nrows
    # gpusAcrossRank = gpus // gpusPerRank
    chunkperloop = gpus * numTbs
    topology = fully_connected(gpus)
    collective = AllReduce(gpus, chunkperloop, True)

    size = chunkperloop

    sizePerTb = size // numTbs
    sizePerRank = sizePerTb // gpus


    with MSCCLPPProgram("hierarchical_allreduce",
        topology,
        collective,
        instances,
        protocol='LL',
        use_double_scratch_buffer=True,
    ):     

        offset = 0
        gpuIds = []
        for n in range(gpus): # collect all GPU Ids in a column
            gpuIds.append( n)
            
        allpairs_reduce_scatter(gpuIds, numTbs, sizePerTb, sizePerRank, 0)

        allpairs_all_gather(gpuIds, numTbs, sizePerTb, sizePerRank, 0, gpus)

        Json()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
# parser.add_argument('num_gpus_per_rank', type=int, help ='number of gpus per rank')
parser.add_argument('num_tbs', type=int, help='number of threadblocks')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()

allreduce(args.num_gpus, args.num_tbs, args.instances)