# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
import json

from msccl.language.types import Buffer, ChannelType, Op, Program, MscclppInstruction as Instruction

_local_src_insts_mscclpp: set = {
    Instruction.put,
    Instruction.put_packet,
    Instruction.signal,
    Instruction.flush,
    Instruction.put_with_signal,
    Instruction.put_with_signal_and_flush,
    Instruction.copy,
    Instruction.copy_packet,
    Instruction.transform_to_packet,
    Instruction.reduce,
    Instruction.reduce_packet,
    Instruction.reduce_send,
    Instruction.reduce_send_packet,
    Instruction.group_load_reduce_store,
    Instruction.group_store,
}
_local_dst_insts_mscclpp: set = {
    Instruction.get,
    Instruction.wait,
    Instruction.read_reduce_copy,
    Instruction.copy,
    Instruction.copy_packet,
    Instruction.transform_to_packet,
    Instruction.reduce,
    Instruction.read_reduce_copy_send,
    Instruction.reduce_send,
    Instruction.reduce_packet,
    Instruction.reduce_send_packet,
    Instruction.group_load_reduce_store,
    Instruction.group_load_reduce,
}

_insts_no_need_sync_barrier: set = {
    Instruction.copy_packet,
    Instruction.reduce_packet,
    Instruction.reduce_send_packet,
    Instruction.barrier,
}


def ir_to_json(program: Program):
    # Figure out sizes of buffers based on usage
    buffer_sizes = defaultdict(lambda: 0)
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                if op.inst in _local_src_insts_mscclpp:
                    key = (gpu.rank, op.src.buffer)
                    buffer_sizes[key] = max(buffer_sizes[key], op.src.index + op.src.size)
                    for src in op.srcs:
                        key = (gpu.rank, src.buffer)
                        buffer_sizes[key] = max(buffer_sizes[key], src.index + src.size)
                if op.inst in _local_dst_insts_mscclpp:
                    key = (gpu.rank, op.dst.buffer)
                    buffer_sizes[key] = max(buffer_sizes[key], op.dst.index + op.dst.size)
                    # ignore remote buffers
                    if (
                        op.inst != Instruction.read_reduce_copy_send
                        and op.inst != Instruction.reduce_send
                        and op.inst != Instruction.reduce_send_packet
                    ):
                        for dst in op.dsts:
                            key = (gpu.rank, dst.buffer)
                            buffer_sizes[key] = max(buffer_sizes[key], dst.index + dst.size)
    for gpu in program.gpus:
        gpu.input_chunks = max(buffer_sizes[(gpu.rank, Buffer.input)], gpu.input_chunks)
        gpu.output_chunks = max(buffer_sizes[(gpu.rank, Buffer.output)], gpu.output_chunks)
        gpu.scratch_chunks = max(buffer_sizes[(gpu.rank, Buffer.scratch)], gpu.scratch_chunks)

    # Since LL protocol will double the scratch size. We need to make sure all GPUs have the same scratch size.
    # Otherwise the offset calculation will be wrong.
    if program.protocol == "LL":
        max_scratch = max(gpu.scratch_chunks for gpu in program.gpus)
        for gpu in program.gpus:
            gpu.scratch_chunks = max_scratch

    # get channel info for each GPU and threadblock
    for gpu in program.gpus:
        gpu.threadblocks = sorted(gpu.threadblocks, key=lambda tb: tb.id)
        chan_dict = {}
        # the channel key is the tuple (srcBuffer, dstBuffer, type)
        for tb in gpu.threadblocks:
            for ch in tb.channels:
                key = (ch.srcBuffer, ch.dstBuffer, ch.type)
                if key not in chan_dict:
                    chan_dict[key] = [(tb.id, ch.connected_to)]
                else:
                    chan_dict[key].append((tb.id, ch.connected_to))
        for key, value in chan_dict.items():
            chan_dict[key] = sorted(value)
        gpu.channels = chan_dict

    # Remove the dependencies of wait after signal. They are actually depends on remote chunk
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                if op.inst == Instruction.wait:
                    op.depends = list(filter(lambda dep: dep.inst != Instruction.signal, op.depends))

    # Filter out redundant dependencies
    # e.g. if op1 and op2 depend on op, and op1 happends before op2
    # then op2 does not need to explicitly depend on op
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            running_depends = []
            for op in tb.ops:
                op.depends = list(filter(lambda dep: dep not in running_depends, op.depends))
                running_depends = running_depends + op.depends

    # Do some additional postprocessing of operations:
    # - Expand operations with dependencies with no-ops
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            new_ops = []
            for op in tb.ops:
                if op.inst in _insts_no_need_sync_barrier:
                    new_ops.append(op)
                    continue
                # Expand extra dependencies into nop operations
                nop = Op(Instruction.nop, -1, None, None, [])
                for i, dep in enumerate(op.depends):
                    # barrier already syncs all threads
                    if dep.inst != Instruction.barrier:
                        nop.depends.append(dep)
                if len(new_ops) > 0 and (
                    new_ops[-1].inst == Instruction.barrier or new_ops[-1].inst == Instruction.nop
                ):
                    new_ops[-1].depends.extend(nop.depends)
                elif len(nop.depends) > 0:
                    new_ops.append(nop)
                new_ops.append(op)
            tb.ops = new_ops

    # update step and tid for ops
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for i, op in enumerate(tb.ops):
                op.step = i
                op.tb = tb.id

    # Need to calculate channel info for each GPU
    nchannels = 0
    for gpu in program.gpus:
        max_tb_channels = 0
        if len(gpu.threadblocks) > 0:
            max_tb_channels = max(tb.channel + 1 for tb in gpu.threadblocks)
        nchannels = max(nchannels, max_tb_channels)
    return dump_to_json(program)


def dump_to_json(program: Program):
    gpus = []

    def get_channel_ids(chunk_list, tb_channel_dict, src_buffer, dst_buffer, chan_type):
        channel_ids = []
        key = (src_buffer, dst_buffer, chan_type)
        if chan_type == ChannelType.nvls:
            ranks = []
            for c in chunk_list:
                ranks.append(c.rank)
            channel_ids.extend(
                [{"id": id} for id, ele in enumerate(tb_channel_dict[key]["connectedTo"]) if set(ele) == set(ranks)]
            )
        else:
            for c in chunk_list:
                channel_ids.extend(
                    [
                        {"id": id, "off": c.index}
                        for id, ele in enumerate(tb_channel_dict[key]["connectedTo"])
                        if ele == c.rank
                    ]
                )
        return channel_ids

    def remove_empty_fields(d):
        return {k: v for k, v in d.items() if v not in [None, "", [], {}]}

    max_scratch = max(gpu.scratch_chunks for gpu in program.gpus)
    max_input = max(gpu.input_chunks for gpu in program.gpus)
    max_output = max(gpu.output_chunks for gpu in program.gpus)

    for id, gpu in enumerate(program.gpus):
        gpu_instance = {
            "id": id,
            "inputChunks": gpu.input_chunks,
            "outputChunks": gpu.output_chunks,
            "scratchChunks": gpu.scratch_chunks,
            "chunkGroups": program.num_chunk_groups,
            "threadblocks": [],
            "channels": [],
        }
        for (srcBuffer, dstBuffer, type), channels in gpu.channels.items():
            obj = {
                "srcbuff": srcBuffer.value if hasattr(srcBuffer, "value") else srcBuffer,
                "dstbuff": dstBuffer.value if hasattr(dstBuffer, "value") else dstBuffer,
                "type": type.value,
                "connectedTo": [eles[1] for eles in channels],
            }
            if type == ChannelType.nvls:
                obj["connectedTo"] = [sorted(list(eles)) for eles in obj["connectedTo"]]
            gpu_instance["channels"].append(obj)
        gpu_instance["channels"] = list(filter(lambda x: x["type"] != "none", gpu_instance["channels"]))
        gpu_instance["channels"] = sorted(gpu_instance["channels"], key=lambda x: (x["srcbuff"], x["dstbuff"]))

        # render for GPU NVLS channels
        for i, chan in enumerate(gpu_instance["channels"]):
            if chan["type"] == "nvls":
                buff = chan["srcbuff"]
                buffer_size = (
                    max_input
                    if buff == Buffer.input.value
                    else max_output if buff == Buffer.output.value else max_scratch
                )
                gpu_instance["channels"][i] = {
                    "buff": chan["srcbuff"],
                    "type": chan["type"],
                    "rankGroups": [{"size": buffer_size, "ranks": ranks} for ranks in chan["connectedTo"]],
                }

        for tb in gpu.threadblocks:
            if tb.id < 0:
                continue
            ops = []
            tb_channels = []
            tb_channel_dict = {}
            for (srcBuffer, dstBuffer, type), channels in gpu.channels.items():
                obj = {
                    "srcbuff": srcBuffer.value if hasattr(srcBuffer, "value") else srcBuffer,
                    "dstbuff": dstBuffer.value if hasattr(dstBuffer, "value") else dstBuffer,
                    "type": type.value,
                    "chanIds": [id for id, ele in enumerate(channels) if ele[0] == tb.id],
                    "connectedTo": [ele[1] for ele in channels if ele[0] == tb.id],
                }
                if len(obj["chanIds"]) > 0:
                    tb_channel_dict[(srcBuffer, dstBuffer, type)] = obj
                    tb_channels.append(obj)
            tb_channels = filter(lambda x: x["type"] != "none", tb_channels)
            tb_channels = sorted(tb_channels, key=lambda x: (x["srcbuff"], x["dstbuff"]))
            for op in tb.ops:
                o_buff = None
                i_buff = None
                dst_channel_ids = []
                src_channel_ids = []
                srcs = []
                dsts = []
                src = None
                dst = None
                if op.tb == -1:
                    continue
                if op.inst == Instruction.signal or op.inst == Instruction.flush:
                    # get dst channel ids
                    dst_channel_ids = get_channel_ids(
                        op.dsts, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
                    )
                    o_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
                elif op.inst == Instruction.wait:
                    # get src channel ids
                    src_channel_ids = get_channel_ids(
                        op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
                    )
                    i_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
                elif op.inst == Instruction.read_reduce_copy:
                    src_channel_ids = get_channel_ids(
                        op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
                    )
                    i_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
                    dst = op.dst
                    src = op.dst  # TODO(binyli): fix this
                elif op.inst == Instruction.read_reduce_copy_send:
                    src_channel_ids = get_channel_ids(
                        op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
                    )
                    dst_channel_ids = get_channel_ids(
                        op.dsts, tb_channel_dict, op.dst.buffer, op.dsts[0].buffer, op.channel_type
                    )
                    i_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
                    o_buff = {"src": op.dst.buffer.value, "dst": op.dsts[0].buffer.value}
                    dst = op.dst
                    src = op.dst  # TODO(binyli): fix this
                elif op.inst == Instruction.reduce_send or op.inst == Instruction.reduce_send_packet:
                    dst_channel_ids = get_channel_ids(
                        op.dsts, tb_channel_dict, op.dst.buffer, op.dsts[0].buffer, ChannelType.sm
                    )
                    o_buff = {"src": op.dst.buffer.value, "dst": op.dsts[0].buffer.value}
                    srcs = list(map(lambda x: {"buff": x.buffer.value, "off": x.index}, op.srcs))
                    dst = op.dst
                    src = op.dst  # TODO(binyli): fix this
                elif op.inst == Instruction.reduce or op.inst == Instruction.reduce_packet:
                    srcs = list(map(lambda x: {"buff": x.buffer.value, "off": x.index}, op.srcs))
                    dst = op.dst
                    src = op.dst
                elif op.inst == Instruction.nop:
                    instr = {
                        "name": op.inst.value,
                        "deps": list(map(lambda dep: {"tb": dep.tb, "step": dep.step}, op.depends)),
                    }
                elif op.inst == Instruction.barrier:
                    instr = {"name": op.inst.value, "nthread_blocks": len(op.extra["tb_list"]), "barrier_id": op.extra["barrier_id"]}
                elif (
                    op.inst == Instruction.put
                    or op.inst == Instruction.put_packet
                    or op.inst == Instruction.put_with_signal
                    or op.inst == Instruction.put_with_signal_and_flush
                ):
                    dst_channel_ids = get_channel_ids(
                        op.dsts, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
                    )
                    o_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
                    srcs = list(map(lambda x: {"buff": x.buffer.value, "off": x.index}, op.srcs))
                elif op.inst == Instruction.get:
                    src_channel_ids = get_channel_ids(
                        op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
                    )
                    i_buff = {"src": op.src.buffer.value, "dst": op.dst.buffer.value}
                    dsts = list(map(lambda x: {"buff": x.buffer.value, "off": x.index}, op.dsts))
                elif (
                    op.inst == Instruction.copy
                    or op.inst == Instruction.copy_packet
                    or op.inst == Instruction.transform_to_packet
                ):
                    src = op.src
                    dst = op.dst
                elif op.inst == Instruction.group_load_reduce_store:
                    src = op.src
                    dst = op.dst
                    src_channel_ids = get_channel_ids(
                        op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
                    )
                    dst_channel_ids = get_channel_ids(
                        op.dsts, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type
                    )
                if op.inst != Instruction.nop and op.inst != Instruction.barrier:
                    instr = {
                        "name": op.inst.value,
                        "i_buff": i_buff,
                        "i_cids": src_channel_ids,
                        "o_buff": o_buff,
                        "o_cids": dst_channel_ids,
                        "src": src.rank if src else None,
                        "srcs": srcs if srcs else None,
                        "dsts": dsts if dsts else None,
                        "srcbuff": src.buffer.value if src and src.buffer else None,
                        "srcoff": src.index if src else None,
                        "dst": dst.rank if dst else None,
                        "dstbuff": dst.buffer.value if dst and dst.buffer else None,
                        "dstoff": dst.index if dst else None,
                        "ctype": op.channel_type.value,
                        "cnt": op.cnt(),
                    }
                ops.append(remove_empty_fields(instr))
            threadblock = {
                "id": tb.id,
                "ops": ops,
                "channels": list(
                    map(
                        lambda x: {"src": x["srcbuff"], "dst": x["dstbuff"], "ctype": x["type"], "cids": x["chanIds"]},
                        tb_channels,
                    )
                ),
            }
            gpu_instance["threadblocks"].append(threadblock)
        gpus.append(gpu_instance)
    obj = {
        "name": program.name,
        "collective": program.collective,
        "protocol": program.protocol,
        "inplace": program.inplace,
        "gpus": gpus,
        "num_threads_per_block": program.num_threads_per_block,
        "use_double_scratch_buffer": program.use_double_scratch_buffer,
        "min_message_size": program.min_message_size,
        "max_message_size": program.max_message_size,
    }
    return json.dumps(obj, indent=2)
