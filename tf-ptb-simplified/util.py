# -*- coding: utf-8 -*-
"""Utilities for Grappler autoparallel optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer


def _replicate_states(metagraph, num_gpus, state_coll_name):
    # CollectionDef型の定義は下記URLの通り（Protocol BuffersのMessage型として定義）
    # - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto
    state_list = metagraph.collection_def[state_coll_name]

    # node_list.valueは変数名の文字列のリストとして定義されている。
    num_states = len(state_list.node_list.value)

    # 各stateを各GPU用に複製してコレクションに追加
    for replica_id in range(1, num_gpus):
        for i in range(num_states):
            orig_name = state_list.node_list.value[i]
            replica_name = "AutoParallel-Replica-{}/{}".format(replica_id, orig_name)
            state_list.node_list.value.append(replica_name)


def _update_snapshot_name(metagraph, var_coll_name):
    # CollectionDef型の定義は下記URLの通り（Protocol BuffersのMessage型として定義）
    # - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto
    var_list = metagraph.collection_def[var_coll_name]

    # 各変数の名称を0番目の複製グラフ用の名前に更新
    for i, value in enumerate(var_list.bytes_list.value):
        var_def = variable_pb2.VariableDef()
        var_def.ParseFromString(value)
        # Somehow node Model/global_step/read doesn't have any fanout and seems to
        # be only used for snapshot; this is different from all other variables.
        if var_def.snapshot_name != "Model/global_step/read:0":
            var_def.snapshot_name = "AutoParallel-Replica-{}/{}".format(0, var_def.snapshot_name)
        value = var_def.SerializeToString()
        var_list.bytes_list.value[i] = value


def auto_parallel(metagraph, num_gpus, model):
    """
    グラフ構造をnum_gpus数のGPU上に自動的に並列化する。
    処理の本体はautoparallel optimizerの実装は下記URLのC++コードの模様。
    グラフをGPUの個数だけ複製して、共有化すべきノード（変数等）を共有化、並列化すべきノード（計算等）を並列化。
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/auto_parallel.cc

    RewriteConfigは下記URLの通りProtocol BuffersのMessage型として定義されており、グラフ構造の最適化指示に用いる。
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/rewriter_config.proto

    なお、autoparallel optimizer以外にもグラフ構造を最適化するoptimizerが用意されているが、今回は使用しない。
    """

    # autoparallel optimizer実行のためのRewriteConfigを設定
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    rewriter_config.optimizers.append("autoparallel")
    rewriter_config.auto_parallel.enable = True
    rewriter_config.auto_parallel.num_replicas = FLAGS.num_gpus

    # 最適化したグラフ定義(optimized_graph)を生成
    optimized_graph = tf_optimizer.OptimizeGraph(rewriter_config, metagraph)

    # metagraphのgraph_defを更新。graph_defはGraphDef型の変数だがこれもProtocol BuffersのMessageとして定義されている。
    # Message::CopyFrom()はMessageをClear()して引数MessageをMerge()することで更新する操作。
    # https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.message#Message.CopyFrom.details
    metagraph.graph_def.CopyFrom(optimized_graph)

    # initial state, final state, variables, trainable_variablesを
    # （多分）最適化したグラフに応じて複製＋名称更新。
    # この必要性はイマイチ理解できていないが…。
    _replicate_states(metagraph, num_gpus, model.initial_state_name)
    _replicate_states(metagraph, num_gpus, model.final_state_name)
    _update_snapshot_name(metagraph, "variables")
    _update_snapshot_name(metagraph, "trainable_variables")
