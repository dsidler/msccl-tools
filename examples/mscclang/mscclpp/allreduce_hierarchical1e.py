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
                peerIdx = peer if peer < r else (peer - 1)
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
                peerIdx = peer if peer < r else (peer - 1)

                if peer != r:
                    c.wait(gpuIds[peer], Buffer.input, chunkIndex, recvtb=tb)

            for peer in range(ngpus):
                peerIdx = peer if peer < r else (peer - 1)
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
def hierarchical_allreduce(gpus, gpusPerRank, instances, protocol):
    nrows = gpusPerRank
    ncols = gpus // nrows
    gpusAcrossRank = gpus // gpusPerRank
    numTbs = 2
    # ncols = gpusPerRank
    # nrows = gpus // ncols
    chunkperloop = gpus * numTbs #gpus * gpus
    topology = fully_connected(gpus)
    collective = AllReduce(gpus, chunkperloop, True)

    innerSize = chunkperloop # // gpusPerRank
    outerSize = chunkperloop // (gpusPerRank) # * gpusAcrossRank)

    # numTbs = 2

    with MSCCLPPProgram("hierarchical_allreduce",
        topology,
        collective,
        instances,
        protocol='Simple',
        # use_double_scratch_buffer=True,
    ):     
        
        # A 4 x 3 GPU arranagement: 4 local GPUs, 3 instances, GPU Ids are numbered as such
        # 0  4   8
        # 1  5   9
        # 2  6   10
        # 3  7  11
        # Reduce-Scatter on each column first, assumption being GPUs in a column have faster connectivity - NVLINK
        # Each GPU exchanges (nrows - 1) * 1/rows of data with other GPUs in the same column 
        # After this step, first GPU in each column will have 1st 1/nrows, 2nd GPU will have 2nd of 1/nrows data reduced
        # size = chunkperloop // nrows
        # size =  innerSize # gpus // gpusPerRank
        sizePerTb = innerSize // numTbs
        sizePerRank = sizePerTb // gpusPerRank
        offset = 0
        for n in range(ncols):
            gpuIds = []
            for m in range(nrows): # collect all GPU Ids in a column
                gpuIds.append( n * nrows + m)
            
            allpairs_reduce_scatter(gpuIds, numTbs, sizePerTb, sizePerRank, 0)

        # for n in range(gpus):
        #     # explicit barrier
        #     r = rank(n)
        #     r.barrier(tb_list=list(range(numTbs))) #TODO assumes

        # Reduce-Scatter across rows, assumption being GPUs in a row have slower connectivity - PCIe, IP NW
        # Each GPU exachanges (1 / rows * cols) * (cols - 1) of data with other GPUs in the same row - less data is exchanged
        # After this step, first GPU each row, will have 1st 1/(nrows * ncols), 2nd will have 2nd of 1/(nrows * ncols)
        offset = sizePerRank #(innerSize // gpusPerRank)
        # sizePerTb = outerSize // numTbs #gpus // (gpusPerRank * gpusAcrossRank) #chunkperloop // (nrows * ncols)
        sizePerRank = outerSize // numTbs // gpusAcrossRank
        for n in range(nrows):
            gpuIds = []
            for m in range(ncols):
                gpuIds.append(n + m * nrows)

            allpairs_reduce_scatter(gpuIds, numTbs, sizePerTb, sizePerRank, offset * n)

        for n in range(gpus):
            # explicit barrier
            r = rank(n)
            r.barrier(tb_list=list(range(numTbs))) #TODO assumes

        # AllGather: AllGather phase goes in reverse order, first gather across rows of GPU
        # After this step, Each GPU in a rows have 1/ncols of data
        for n in range(nrows):
            gpuIds = []
            for m in range(ncols):
                gpuIds.append(n + m * nrows)

            allpairs_all_gather(gpuIds, numTbs, sizePerTb, sizePerRank, offset * n)

        for n in range(gpus):
            # explicit barrier
            r = rank(n)
            r.barrier(tb_list=list(range(numTbs))) #TODO assumes

        # AllGather: AllGather phase goes in reverse order, 2nd AllGather across columns of GPU
        # After this step, Each GPU the systems will have complete reduced data
        # size = innerSize # gpus // gpusPerRank #chunkperloop // nrows
        sizePerRank = sizePerTb // gpusPerRank
        offset = 0
        for n in range(ncols):
            gpuIds = []
            for m in range(nrows):
                gpuIds.append( n * nrows + m)

            allpairs_all_gather(gpuIds, numTbs, sizePerTb, sizePerRank, 0)

        Json()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('num_gpus_per_rank', type=int, help ='number of gpus per rank')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

hierarchical_allreduce(args.num_gpus, args.num_gpus_per_rank, args.instances, args.protocol)