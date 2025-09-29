import itertools
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import polars as pl
from google.protobuf.json_format import ParseDict

import constants
from timecomputation_reproduce import TimeComputation


warnings.simplefilter(action="ignore", category=Warning)

def dict_to_proto_with_defaults(data: dict, message_cls, *, init_nested=False, ignore_unknown=True):
    msg = message_cls()
    if init_nested:
        _init_all_submessages(msg)  # see helper below
    ParseDict(data, msg, ignore_unknown_fields=ignore_unknown)
    return msg

def _init_all_submessages(msg):
    # Walk descriptor; instantiate any unset message fields (recursively).
    desc = msg.DESCRIPTOR
    for field in desc.fields:
        if field.label == field.LABEL_REPEATED:
            continue  # repeated fields default to empty; nothing to init
        if field.cpp_type == field.CPPTYPE_MESSAGE:
            # Make sure the submessage exists (is "present")
            sub = getattr(msg, field.name)
            sub.SetInParent()  # marks present, fills its scalar defaults
            _init_all_submessages(sub)





        
class ProcessMapBuilder:
    def __init__(self, process_list, same_shape):
        self.process_list = process_list
        self.same_shape = same_shape
        self.PROCESSING_TIME_LIST = {}
        self.DEPEND_DICT = {}
        self.COMPLETE_PROCESS_ID = []
        self.DEPENT_FORWARD = {}
        self.FIXED_DICT_PROCESS = {}

        for p in self.process_list:
            self.FIXED_DICT_PROCESS[p['process_id']] = p['fixed']

    @staticmethod
    def build_existed_process_list(list_processed_process):
        list_process_id = []
        for item in list_processed_process:
            for x in item['process_id']:
                list_process_id.extend(x)
        return list_process_id

    def build_process_graph_root(self) -> List:
        process_graph = []
        layer = 1
        def build_process_graph(process_list, process_graph, layer):
            layer += 1
            list_existed = self.build_existed_process_list(process_graph)
            process_graph_tmp = {}
            list_process_match = []
            for process in process_list:
                if all(item in list_existed for item in process['depend_on']):
                    list_process_match.append(process['process_id'])
            list_process_match = [tuple(list_process_match)]
            process_graph_tmp['layout'] = layer
            process_graph_tmp['process_id'] = list_process_match
            process_graph.append(process_graph_tmp)
            new_process_list = [process for process in process_list if process['process_id'] not in self.build_existed_process_list(process_graph)]
            if len(new_process_list) != 0:
                build_process_graph(new_process_list, process_graph, layer)
            return new_process_list, process_graph

        process_graph_tmp = {}
        if self.same_shape:
            process_match = defaultdict(list)
        else:
            process_match = []
        list_process_fixed = []
        for process in self.process_list:
            if process['fixed'] and process["product_shape"]['shape'] == constants.OTHER_PRODUCT_SHAPE_NAME['shape']:
                list_process_fixed.append(process['process_id'])
            elif process['depend_on'] is None:
                if self.same_shape:
                    shape = process['product_shape'].get("shape", False)
                    dims = process['product_shape'].get("dimension", False)
                    if shape and dims:
                        dims = [int(x) for x in dims.split("x")]
                        keys = (constants.OTHER_PRODUCT_SHAPE_NAME['shape'])
                        if shape == "ROUND" and dims[0]==dims[1]:
                            keys = (("ROUND", dims[0]))
                        elif shape == "ANGLE":
                            dims = list(map(int, dims))
                            pairs = itertools.combinations(dims, 2)
                            keys = tuple([("ANGLE", tuple(sorted(p))) for p in pairs])
                        elif shape == "PLATE":
                            keys = (constants.OTHER_PRODUCT_SHAPE_NAME['shape'])
                        else:
                            keys = (constants.OTHER_PRODUCT_SHAPE_NAME['shape'])
                    else:
                        keys = (constants.OTHER_PRODUCT_SHAPE_NAME['shape'])
                    for key in keys:
                        process_match[key].append(process)
                else:
                    process_match.append(process['process_id'])
        if self.same_shape:
            for shape_info in process_match:
                process_match[shape_info].sort(key=lambda x: x["final_deadline"])
            process_match = [tuple(k['process_id'] for k in v) for v in process_match.values()]
            process_match = sorted(process_match, key=lambda x: (-len(x), process_match.index(x)))
        else:
            process_match = [tuple(process_match)]
        process_graph_tmp['layout'] = 0
        process_graph_tmp['process_id'] = [tuple(list_process_fixed)]
        process_graph.append(process_graph_tmp)

        process_graph_tmp = {}
        process_graph_tmp['layout'] = layer
        process_graph_tmp['process_id'] = process_match
        process_graph.append(process_graph_tmp)
        new_process_list = [process for process in self.process_list if process['process_id'] not in self.build_existed_process_list(process_graph)]
        if len(new_process_list) != 0:
            build_process_graph(new_process_list, process_graph, layer)
        return process_graph

    def build_dependency(self) -> None:
        for p in self.process_list:
            self.PROCESSING_TIME_LIST[p['process_id']] = p['machining_time']
            if p['depend_on'] is not None:
                for i in p['depend_on']:
                    if i not in self.DEPEND_DICT:
                        self.DEPEND_DICT[i] = [p['process_id']]
                    else:
                        self.DEPEND_DICT[i].append(p['process_id'])
            if p['fixed']:
                self.COMPLETE_PROCESS_ID.append(p['process_id'])

        # Now, populate the next_process_ids for each process
        for process in self.process_list:
            process_id = process['process_id']
            self.DEPENT_FORWARD[process_id] = []

            # Create a set to keep track of visited processes to avoid cycles
            visited = set()

            def find_next_processes(current_id):
                if current_id in visited:
                    return
                visited.add(current_id)
                self.DEPENT_FORWARD[process_id].append(current_id)
                if current_id in list(self.DEPEND_DICT.keys()):
                    for next_id in self.DEPEND_DICT[current_id]:
                        find_next_processes(next_id)
            find_next_processes(process_id)


if __name__ == "__main__":
    pass