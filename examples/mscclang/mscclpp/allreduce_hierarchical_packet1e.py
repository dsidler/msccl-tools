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
def hierarchical_allreduce(gpus, gpusPerRank, numTbs, instances):
    # nrows = gpusPerRank
    # ncols = gpus // nrows
    gpusAcrossRank = gpus // gpusPerRank
    chunkperloop = gpus * numTbs
    topology = fully_connected(gpus)
    collective = AllReduce(gpus, chunkperloop, True)

    innerSize = chunkperloop # // gpusPerRank
    outerSize = chunkperloop // (gpusPerRank) # * gpusAcrossRank)


    with MSCCLPPProgram("hierarchical_allreduce",
        topology,
        collective,
        instances,
        protocol='LL',
        use_double_scratch_buffer=True,
    ):     
        
        # A 4 x 3 GPU arranagement: 4 local GPUs, 3 instances, GPU Ids are numbered as such
        # 0  4   8
        # 1  5   9
        # 2  6   10
        # 3  7  11
        # Reduce-Scatter on each column first, assumption being GPUs in a column have faster connectivity - NVLINK
        # Each GPU exchanges (nrows - 1) * 1/rows of data with other GPUs in the same column 
        # After this step, first GPU in each column will have 1st 1/nrows, 2nd GPU will have 2nd of 1/nrows data reduced
        sizePerTb = innerSize // numTbs
        sizePerRank = sizePerTb // gpusPerRank
        offset = 0
        for n in range(gpusAcrossRank):
            gpuIds = []
            for m in range(gpusPerRank): # collect all GPU Ids in a column
                gpuIds.append( n * gpusPerRank + m)
            
            allpairs_reduce_scatter(gpuIds, numTbs, sizePerTb, sizePerRank, 0)

        # Reduce-Scatter across rows, assumption being GPUs in a row have slower connectivity - PCIe, IP NW
        # Each GPU exachanges (1 / rows * cols) * (cols - 1) of data with other GPUs in the same row - less data is exchanged
        # After this step, first GPU each row, will have 1st 1/(nrows * ncols), 2nd will have 2nd of 1/(nrows * ncols)
        offset = sizePerRank
        # size = chunkperloop // (nrows * ncols)
        sizePerRank = outerSize // numTbs // gpusAcrossRank
        for n in range(gpusPerRank):
            gpuIds = []
            for m in range(gpusAcrossRank):
                gpuIds.append(n + m * gpusPerRank)

            allpairs_reduce_scatter(gpuIds, numTbs, sizePerTb, sizePerRank, offset * n)

        # AllGather: AllGather phase goes in reverse order, first gather across rows of GPU
        # After this step, Each GPU in a rows have 1/ncols of data
        for n in range(gpusPerRank):
            gpuIds = []
            for m in range(gpusAcrossRank):
                gpuIds.append(n + m * gpusPerRank)

            allpairs_all_gather(gpuIds, numTbs, sizePerTb, sizePerRank, offset * n, chunkperloop)

        # AllGather: AllGather phase goes in reverse order, 2nd AllGather across columns of GPU
        # After this step, Each GPU the systems will have complete reduced data
        # size = chunkperloop // nrows
        sizePerRank = sizePerTb // gpusPerRank
        offset = 0
        for n in range(gpusAcrossRank):
            gpuIds = []
            for m in range(gpusPerRank):
                gpuIds.append( n * gpusAcrossRank + m)

            allpairs_all_gather(gpuIds, numTbs, sizePerTb, sizePerRank, 0, gpus)

        Json()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('num_gpus_per_rank', type=int, help ='number of gpus per rank')
parser.add_argument('num_tbs', type=int, help='number of threadblocks')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()

hierarchical_allreduce(args.num_gpus, args.num_gpus_per_rank, args.num_tbs, args.instances)