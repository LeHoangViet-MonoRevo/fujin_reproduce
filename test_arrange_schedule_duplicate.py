from collections import defaultdict
from datetime import datetime
from pprint import pprint

import pandas as pd
import polars as pl

from arrange_schedule_v2_reproduce import ArrangeScheduleService
from constants import constants
from processmapbuilder_reproduce import ProcessMapBuilder

# =================================================================================
# 1. DEFINE INPUTS
# =================================================================================

# --- Basic Factory & Schedule Info ---
start_time = "2025-10-02 08:00:00"
end_time = "2025-10-03 17:00:00"

# --- List of Processes to Schedule ---
# This is the main input describing the jobs to be done.
process_list = [
    {
        "process_id": "lot1___10",
        "timing_details": {
            "pre_processing": 15,
            "processing": 60,
            "post_processing": 15,
        },
        "machining_time": 90,
        "depend_on": [],  # No dependencies
        "is_internal": True,
        "process_category": "cutting",
        "worker_group": "group_a",
        "worker_id": None,
        "machine_id": None,
        "fixed": False,
        "is_work_break_time": False,
        "overtime": False,
        "is_nightshift_expected": False,
        "linker_id": None,
        "final_deadline": "2025-10-02 17:00:00",
        "material_deadline": "",
        "product_shape": {"shape": "A"},
        "status": 1,
        "actual_start_at": None,
    },
    {
        "process_id": "lot1___20",
        "timing_details": {
            "pre_processing": 10,
            "processing": 90,
            "post_processing": 10,
        },
        "machining_time": 110,
        "depend_on": ["lot1___10"],  # Must start after Task 10
        "is_internal": True,
        "process_category": "welding",
        "worker_group": "group_a",
        "worker_id": None,
        "machine_id": None,
        "fixed": False,
        "is_work_break_time": False,
        "overtime": False,
        "is_nightshift_expected": False,
        "linker_id": None,
        "final_deadline": "2025-10-02 17:00:00",
        "material_deadline": "",
        "product_shape": {"shape": "A"},
        "status": 1,
        "actual_start_at": None,
    },
]

# --- Factory Resource Information ---
# Describes machines, workers, and their skills/capabilities.
factory_rs_info = {
    "dict_process_category_to_machine": {"cutting": [101], "welding": [102]},
    "dict_machine_to_worker": {
        101: [1],  # Machine 101 can be operated by Worker 1
        102: [1],  # Machine 102 can also be operated by Worker 1
    },
    "dict_worker_group_to_worker": {"group_a": [1]},
    "dict_machine_auto": {101: False, 102: False},
    "list_machine_active": [101, 102],
}

# --- DataFrames with existing schedules ---
worker_details_df = pd.DataFrame({"worker_id": [1]})
machine_details_df = pd.DataFrame({"machine_id": [101, 102]})
process_df = pd.DataFrame({"final_deadline": [datetime(2025, 10, 2, 17, 0, 0)]})

# Let's say Worker 1 is already busy from 9:00 to 10:00 AM on another task
working_time_available_worker_df = pd.DataFrame(
    {
        "start_date": [datetime(2025, 10, 2, 9, 0, 0)],
        "end_date": [datetime(2025, 10, 2, 10, 0, 0)],
        "worker_id": [1],
        "process_id": ["some_other_task"],
        "machine_id": [999],
        "processing_time": [60],
    }
)

# For this example, the machines are completely free.
working_time_available_machine_df = pd.DataFrame(
    columns=["start_date", "end_date", "machine_id", "process_id", "processing_time"]
)

# =================================================================================
# 2. INSTANTIATE THE SERVICE AND RUN THE SCHEDULER
# =================================================================================

# -- Initialize the main service class --
scheduler = ArrangeScheduleService(
    working_hour_start=8,
    working_mins_start=0,
    working_hour_end=17,
    working_mins_end=0,
    official_work_days=[0, 1, 2, 3, 4],
    morning_end_time_hours=12,
    morning_end_time_mins=0,
    afternoon_start_time_hours=13,
    afternoon_start_time_mins=0,
    overtime_start_time_hours=17,
    overtime_start_time_mins=0,
    overtime_end_time_hours=20,
    overtime_end_time_mins=0,
    holiday={},
    list_breaktime=[],
    process_list=process_list,
    start_planning_time=start_time,
    end_planning_time=end_time,
    settings=type("Settings", (), {"same_shape": False}),
    factory_rs_info=factory_rs_info,
    customer_rs={},
    timezone=None,
)

# Replace the class's process map builder with our mock one for this test
scheduler.process_map_object = ProcessMapBuilder(process_list, False)
scheduler.process_map_object.build_dependency()


# -- Call the main scheduling method --
final_schedule = scheduler.build_schedule(
    worker_details_df,
    working_time_available_worker_df,
    machine_details_df,
    working_time_available_machine_df,
    process_df,
)

# =================================================================================
# 3. PRINT THE FINAL SCHEDULE
# =================================================================================
print("--- FINAL SCHEDULE ---")
pprint(final_schedule)
