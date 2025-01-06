# Copyright (c) 2024 Advanced Micro Devices.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allpairs_reduce_scatter(gpuIds, size, offset, dataSize):
    ngpus = len(gpuIds)

    for r in range(ngpus):
        for index in range(size):
            chunkIndex = offset + r * size + index
            tb = index % ngpus
            c = chunk(gpuIds[r], Buffer.input, chunkIndex)

            for peer in range(ngpus):
                peerIdx = peer if peer < r else (peer - 1)
                if r != peer:
                    peerChunkIndex = offset + peer * size + index
                    # c = chunk(gpuIds[r], Buffer.input, chunkIndex) #, size)
                    c.signal(gpuIds[peer], Buffer.input, peerChunkIndex, sendtb=tb)

            for peer in range(ngpus):
                peerIdx = peer if peer < r else (peer - 1)

                if peer != r:
                    c.wait(gpuIds[peer], Buffer.input, chunkIndex, recvtb=tb)

            for peer in range(ngpus):
                peerIdx = peer if peer < r else (peer - 1)
                if peer != r:
                    c.reduce(chunk(gpuIds[peer], Buffer.input, chunkIndex), recvtb=tb)
 




# def allpairs_reduce_scatter_put(gpuIds, size, offset, dataSize):
#     ngpus = len(gpuIds)



def allpairs_all_gather(gpuIds, size, offset, dataSize):
    ngpus = len(gpuIds)

    for r in range(ngpus):
        for index in range(size):
            tb = index % ngpus
            chunkIndex = offset + r * size + index
            c = chunk(gpuIds[r], Buffer.input, chunkIndex)#, size)
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

                peerChunkIndex = offset + peer * size + index
                c = chunk(gpuIds[r], Buffer.input, peerChunkIndex)#, size)
                if peer != r:
                    c.wait(gpuIds[peer], Buffer.input, peerChunkIndex, recvtb=tb)

# Performs two levels of allReduce
def hierarchical_allreduce(gpus, gpusPerRank, instances, protocol):
    nrows = gpusPerRank
    ncols = gpus // nrows
    gpusAcrossRank = gpus // gpusPerRank
    # ncols = gpusPerRank
    # nrows = gpus // ncols
    chunkperloop = gpus #gpus * gpus
    topology = fully_connected(gpus)
    collective = AllReduce(gpus, chunkperloop, True)

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
        size = gpus // gpusPerRank
        offset = 0
        for n in range(ncols):
            gpuIds = []
            for m in range(nrows): # collect all GPU Ids in a column
                gpuIds.append( n * nrows + m)
            
            allpairs_reduce_scatter(gpuIds, size, 0, gpus)

        for n in range(gpus):
            # explicit barrier
            r = rank(n)
            r.barrier(tb_list=list(range(gpusPerRank))) #TODO assumes

        # Reduce-Scatter across rows, assumption being GPUs in a row have slower connectivity - PCIe, IP NW
        # Each GPU exachanges (1 / rows * cols) * (cols - 1) of data with other GPUs in the same row - less data is exchanged
        # After this step, first GPU each row, will have 1st 1/(nrows * ncols), 2nd will have 2nd of 1/(nrows * ncols)
        offset = size
        size = gpus // (gpusPerRank * gpusAcrossRank) #chunkperloop // (nrows * ncols)
        for n in range(nrows):
            gpuIds = []
            for m in range(ncols):
                gpuIds.append(n + m * nrows)

            allpairs_reduce_scatter(gpuIds, size, offset * n, gpus)
            # allpairs_reduce_scatter_put(gpuIds, size, offset * n, gpus)

        for n in range(gpus):
            # explicit barrier
            r = rank(n)
            r.barrier(tb_list=list(range(gpusAcrossRank))) #TODO assumes

        # AllGather: AllGather phase goes in reverse order, first gather across rows of GPU
        # After this step, Each GPU in a rows have 1/ncols of data
        for n in range(nrows):
            gpuIds = []
            for m in range(ncols):
                gpuIds.append(n + m * nrows)

            allpairs_all_gather(gpuIds, size, offset * n, chunkperloop)

        for n in range(gpus):
            # explicit barrier
            r = rank(n)
            r.barrier(tb_list=list(range(gpusAcrossRank))) #TODO assumes

        # AllGather: AllGather phase goes in reverse order, 2nd AllGather across columns of GPU
        # After this step, Each GPU the systems will have complete reduced data
        size = gpus // gpusPerRank #chunkperloop // nrows
        offset = 0
        for n in range(ncols):
            gpuIds = []
            for m in range(nrows):
                gpuIds.append( n * nrows + m)

            allpairs_all_gather(gpuIds, size, 0, gpus)

        Json()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('num_gpus_per_rank', type=int, help ='number of gpus per rank')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

hierarchical_allreduce(args.num_gpus, args.num_gpus_per_rank, args.instances, args.protocol)