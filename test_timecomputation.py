from collections import defaultdict
from datetime import datetime

from constants import constants
from timecomputation_reproduce import TimeComputation, TimeComputationExtention

# Step 1: Initialise the scheduling class
time_computer = TimeComputationExtention(
    working_hour_start=8,
    working_mins_start=0,
    working_hour_end=17,
    working_mins_end=0,
    official_work_days=[0, 1, 2, 3, 4],  # Monday to Friday
    morning_end_time_hours=12,
    morning_end_time_mins=0,
    afternoon_start_time_hours=13,
    afternoon_start_time_mins=0,
    overtime_start_time_hours=17,
    overtime_start_time_mins=0,
    overtime_end_time_hours=20,
    overtime_end_time_mins=0,
    holiday={"days": ["2025-10-03"]},  # A Friday holiday
    list_breaktime=[((10, 0), (10, 15))],  # 15-min tea break
    factory_rs_info={
        # Machine 101 is manual, 102 is automated
        "dict_machine_auto": {101: False, 102: True},
        # Not used in this specific method call, but good practice to include
        "dict_deadline_fixed": {},
    },
    customer_rs={},
    timezone=None,
)

# Set necessary attributes not included in the constructor
time_computer.dt_current_date = datetime(2025, 10, 1)
time_computer.dict_total_worked_time = {
    "worker": defaultdict(int),
    "machine": defaultdict(int),
}


# Step 2: Define the inputs for the `schedule_core_linker` method
process_unit_to_schedule = {
    "timing_details": {
        "pre_processing": 30,  # 30 mins
        "processing": 90,  # 90 mins
        "post_processing": 15,  # 15 mins
    },
    "is_work_break_time": False,  # Task must pause for lunch/tea breaks
    "is_nightshift_expected": False,
}

child_task_schedule = {
    # Key format: "worker_id___machine_id___group_index"
    "1___101___0": {
        "allocated_time_dict": {
            "pre_processing": [
                (datetime(2025, 10, 1, 8, 30), datetime(2025, 10, 1, 9, 0))
            ],
            "processing": [
                (datetime(2025, 10, 1, 9, 0), datetime(2025, 10, 1, 10, 0)),
                (
                    datetime(2025, 10, 1, 10, 15),
                    datetime(2025, 10, 1, 11, 15),
                ),  # Paused for tea break
            ],
            "post_processing": [
                (datetime(2025, 10, 1, 11, 15), datetime(2025, 10, 1, 11, 45))
            ],
        },
        "end": datetime(2025, 10, 1, 11, 45),
        # ...other metadata
    }
}

worker_availability = {
    # Key format: "worker_id___machine_id"
    "1___101": [(datetime(2025, 10, 1, 8, 0), datetime(2025, 10, 1, 17, 0))]
}
machine_availability = {
    # Key format: machine_id
    101: [(datetime(2025, 10, 1, 8, 0), datetime(2025, 10, 1, 17, 0))]
}

deadline = datetime(2025, 10, 1, 17, 0)

print("Scheduling Task B, which must start after Task A finishes at 11:45 AM...\n")

decision_result = time_computer.schedule_core_linker(
    process_unit=process_unit_to_schedule,
    decision_dict_child=child_task_schedule,
    factory_resources=worker_availability,
    deadline=deadline,
    overtime=False,
    night_time_prioritize=False,
    keep_deadline=True,
    time_grouper_machine=machine_availability,
    force_start=None,
)


# --- Print the output in a readable format ---
if not decision_result:
    print("❌ No valid schedule found.")
else:
    # The method now returns a dictionary, so we access its items
    for resource_id, schedule_info in decision_result.items():
        worker_id, machine_id, _ = resource_id.split("___")
        print(
            f"✅ Optimal Schedule Found for Worker {worker_id} on Machine {machine_id}:"
        )

        schedule = schedule_info["allocated_time_dict"]
        start_time = schedule["pre_processing"][0][0]
        end_time = schedule["post_processing"][-1][1]

        print(f"   -> Start Time: {start_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"   -> End Time:   {end_time.strftime('%Y-%m-%d %H:%M')}")

        print("\n   Detailed Breakdown:")
        pre_start = schedule["pre_processing"][0][0]
        pre_end = schedule["pre_processing"][-1][1]
        print(
            f"   - Pre-processing:  {pre_start.strftime('%H:%M')} -> {pre_end.strftime('%H:%M')}"
        )

        proc_start = schedule["processing"][0][0]
        proc_end = schedule["processing"][-1][1]
        print(
            f"   - Processing:      {proc_start.strftime('%H:%M')} -> {proc_end.strftime('%H:%M')} (Note: Pauses for lunch at 12:00, resumes at 13:00)"
        )

        post_start = schedule["post_processing"][0][0]
        post_end = schedule["post_processing"][-1][1]
        print(
            f"   - Post-processing: {post_start.strftime('%H:%M')} -> {post_end.strftime('%H:%M')}"
        )
        break  # Only print the first (best) result
