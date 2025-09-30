import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import polars as pl
from google.protobuf.json_format import ParseDict

from constants import constants

warnings.simplefilter(action="ignore", category=Warning)


def dict_to_proto_with_defaults(
    data: dict, message_cls, *, init_nested=False, ignore_unknown=True
):
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


class TimeComputation:
    def __init__(
        self,
        working_hour_start,
        working_mins_start,
        working_hour_end,
        working_mins_end,
        official_work_days,
        morning_end_time_hours,
        morning_end_time_mins,
        afternoon_start_time_hours,
        afternoon_start_time_mins,
        overtime_end_time_hours,
        overtime_end_time_mins,
        overtime_start_time_hours,
        overtime_start_time_mins,
        holiday,
        list_breaktime,
        factory_rs_info,
        customer_rs,
        timezone,
        base=None,
        request_id=None,
        organization_id=None,
    ):
        self.WORKING_HOURS_START = time(working_hour_start, working_mins_start)
        self.WORKING_HOURS_END = time(working_hour_end, working_mins_end)
        self.WORKING_DAYS = official_work_days
        self.MORNING_END_TIME = time(morning_end_time_hours, morning_end_time_mins)
        self.AFTERNOON_START_TIME = time(
            afternoon_start_time_hours, afternoon_start_time_mins
        )
        self.OVERTIME_END_TIME = time(overtime_end_time_hours, overtime_end_time_mins)
        self.OVERTIME_START_TIME = time(
            overtime_start_time_hours, overtime_start_time_mins
        )
        self.holiday = holiday
        list_breaktime = [
            (time(s[0], s[1]), time(e[0], e[1])) for s, e in list_breaktime
        ] + [(self.MORNING_END_TIME, self.AFTERNOON_START_TIME)]
        list_breaktime = sorted((s, e) for s, e in list_breaktime if s != e)
        self.list_breaktime = []
        for s, e in list_breaktime:
            if self.list_breaktime and s <= self.list_breaktime[-1][1]:
                self.list_breaktime[-1] = (
                    self.list_breaktime[-1][0],
                    max(self.list_breaktime[-1][1], e),
                )
            else:
                self.list_breaktime.append((s, e))
        self.customer_rs = customer_rs

        self.WORKING_HOUR_RANGE = [(self.WORKING_HOURS_START, self.WORKING_HOURS_END)]
        self.WORKING_HOURS_END_FIXED = time(working_hour_end, working_mins_end)
        self.holiday_fixed = holiday
        self.WORKING_DAYS_FIXED = official_work_days

        self.hours_of_break = (
            (self.AFTERNOON_START_TIME.hour * 60 + self.AFTERNOON_START_TIME.minute)
            - (self.MORNING_END_TIME.hour * 60 + self.MORNING_END_TIME.minute)
        ) / 60
        self.hours_in_day = (
            (self.WORKING_HOURS_END.hour * 60 + self.WORKING_HOURS_END.minute)
            - (self.WORKING_HOURS_START.hour * 60 + self.WORKING_HOURS_START.minute)
        ) / 60
        self.hours_in_day_overtime = (
            (self.OVERTIME_END_TIME.hour * 60 + self.OVERTIME_END_TIME.minute)
            - (self.OVERTIME_START_TIME.hour * 60 + self.OVERTIME_START_TIME.minute)
        ) / 60
        self.factory_rs_info = factory_rs_info

        self.base_id = base
        self.request_id = request_id
        self.organization_id = organization_id

    @contextmanager
    def adjust_schedule_time(self, overtime=False):
        if overtime:
            overrides = {
                "WORKING_DAYS": self.WORKING_DAYS,
                "holiday": self.holiday,
                "WORKING_HOURS_START": self.OVERTIME_START_TIME,
                "WORKING_HOURS_END": self.OVERTIME_END_TIME,
                "WORKING_HOUR_RANGE": [
                    (self.OVERTIME_START_TIME, self.OVERTIME_END_TIME)
                ],
            }
            original = {k: getattr(self, k) for k in overrides}
            try:
                for k, v in overrides.items():
                    setattr(self, k, v)
                yield
            finally:
                for k, v in original.items():
                    setattr(self, k, v)
        else:
            yield

    def get_batches(self, total_tasks: int, batch_size: int = 200) -> list[int]:
        if total_tasks <= 0 or batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")
        batches = []
        current_size = 0
        while current_size < total_tasks:
            current_size += batch_size
            if current_size > total_tasks:
                batches.append(total_tasks)
                break
            batches.append(current_size)
        return batches

    def count_days_off_func(self, start, end):
        ### Counts holidays but removes the first and last days
        if isinstance(start, str) and isinstance(end, str):
            current_datetime = datetime.strptime(start, constants.DT_FORMAT)
            delivery_datetime = datetime.strptime(end, constants.DT_FORMAT)
        else:
            current_datetime = start
            delivery_datetime = end
        all_days = []
        dt_current_date_temp = current_datetime.replace(minute=0)
        while dt_current_date_temp.date() <= delivery_datetime.date():
            all_days.append(dt_current_date_temp)
            dt_current_date_temp += timedelta(days=1)
        middle_dates = all_days[1:-1]
        middle_dates = [
            x
            for x in middle_dates
            if x.weekday() not in self.WORKING_DAYS or str(x.date()) in self.holiday
        ]
        count_days_off = len(middle_dates)
        return count_days_off

    def is_day_off(self, day):
        if str(day.date()) in self.holiday.get("working_days", []):
            return False
        if str(day.date()) in self.holiday.get("days", []):
            return True
        if str(day.year) in self.holiday.get("years", []):
            return True
        if str(day.month) in self.holiday.get("months", []):
            return True
        if day.weekday() not in self.WORKING_DAYS:
            return True
        return False

    def is_day_off_external(self, day, organizations: int):
        holiday_by_organizations = self.customer_rs[organizations]
        if str(day.date()) in holiday_by_organizations["working_days"]:
            return True
        if day.year in holiday_by_organizations["year_holidays"]:
            return True
        if day.month in holiday_by_organizations["month_holidays"]:
            return True
        if day.weekday() in holiday_by_organizations["weekly_holidays"]:
            return True
        if str(day.date()) in holiday_by_organizations["holidays"]:
            return True
        return False

    def is_all_day_off(self, start_date, end_date):
        if start_date.date() == end_date.date() - timedelta(days=1):
            return True
        cur_date = start_date + timedelta(days=1)
        while cur_date <= end_date - timedelta(days=1):
            if not self.is_day_off(cur_date):
                return False
            cur_date += timedelta(days=1)
        return True

    def find_overlap(self, A_start, A_end, B_start, B_end, compute_hours=False):
        overlap_start = max(A_start, B_start)
        overlap_end = min(A_end, B_end)
        if overlap_start < overlap_end:
            if not compute_hours:
                return overlap_start, overlap_end
            overlap = overlap_end - overlap_start
            return overlap.days * 24 + overlap.seconds / 3600
        else:
            return 0

    def find_max_deadline_allow(self, pid, subkey):
        subdict = self.factory_rs_info["dict_deadline_fixed"].get(pid, {})
        if not subdict:
            return None
        sorted_keys = sorted(subdict.keys())
        for k in sorted_keys:
            if k >= subkey:
                return subdict[k]
        return None

    def max_end_date_in_dict(self, allocated_time_dict: dict):
        max_end = None
        for time_list in allocated_time_dict.values():
            for start, end in time_list:
                if max_end is None or end > max_end:
                    max_end = end
        return max_end

    def min_start_date_in_dict(self, allocated_time_dict: dict):
        min_start = None
        for time_list in allocated_time_dict.values():
            for start, end in time_list:
                if min_start is None or start < min_start:
                    min_start = start
        return min_start

    def compute_night_time_hours(self, schedule):
        df = pd.DataFrame(schedule, columns=["start", "end"])

        def clip_range(row):
            date = row["start"].date()
            before_work = max(
                timedelta(0),
                min(row["end"], datetime.combine(date, time(8, 0))) - row["start"],
            )
            after_work = max(
                timedelta(0),
                row["end"] - max(row["start"], datetime.combine(date, time(17, 0))),
            )
            return (before_work + after_work).total_seconds() / 3600

        return df.apply(clip_range, axis=1).sum()

    def compute_lunch_break(self, start, end):
        total_time_lunch_break = 0
        if isinstance(start, str) and isinstance(end, str):
            current_datetime = datetime.strptime(start, constants.DT_FORMAT)
            delivery_datetime = datetime.strptime(end, constants.DT_FORMAT)
        else:
            current_datetime = start
            delivery_datetime = end
        current_datetime_lunch_start = current_datetime.replace(
            hour=self.MORNING_END_TIME.hour, minute=self.MORNING_END_TIME.minute
        )
        current_datetime_lunch_end = current_datetime.replace(
            hour=self.AFTERNOON_START_TIME.hour, minute=self.AFTERNOON_START_TIME.minute
        )
        delivery_datetime_lunch_start = delivery_datetime.replace(
            hour=self.MORNING_END_TIME.hour, minute=self.MORNING_END_TIME.minute
        )
        delivery_datetime_lunch_end = delivery_datetime.replace(
            hour=self.AFTERNOON_START_TIME.hour, minute=self.AFTERNOON_START_TIME.minute
        )
        if current_datetime.date() == delivery_datetime.date() and not self.is_day_off(
            current_datetime
        ):
            total_time_lunch_break += self.find_overlap(
                current_datetime,
                delivery_datetime,
                delivery_datetime_lunch_start,
                current_datetime_lunch_end,
                True,
            )
        else:
            diff_date = delivery_datetime.replace(
                hour=0, minute=0
            ) - current_datetime.replace(hour=0, minute=0)
            all_days = []
            dt_current_date_temp = current_datetime.replace(minute=0)
            while dt_current_date_temp.date() <= delivery_datetime.date():
                all_days.append(dt_current_date_temp)
                dt_current_date_temp += timedelta(days=1)
            middle_dates = all_days[1:-1]
            middle_dates = [x for x in middle_dates if self.is_day_off(x)]
            count_days_off = len(middle_dates)
            if diff_date.days >= 2:
                total_time_lunch_break = (
                    total_time_lunch_break
                    + ((diff_date.days - 1) * self.hours_of_break)
                    - self.hours_of_break * count_days_off
                )

            if current_datetime <= current_datetime_lunch_end and not self.is_day_off(
                current_datetime
            ):
                total_time_lunch_break += (
                    current_datetime_lunch_end
                    - max(current_datetime, current_datetime_lunch_start)
                ).total_seconds() / 3600
            if (
                delivery_datetime >= delivery_datetime_lunch_start
                and not self.is_day_off(delivery_datetime)
            ):
                total_time_lunch_break += (
                    min(delivery_datetime_lunch_end, delivery_datetime)
                    - delivery_datetime_lunch_start
                ).total_seconds() / 3600
        return total_time_lunch_break

    def get_diff_hours(self, start, end, is_work_break_time=False) -> int:
        hours_working_in_day = (
            (self.WORKING_HOURS_END.hour * 60 + self.WORKING_HOURS_END.minute)
            - (self.WORKING_HOURS_START.hour * 60 + self.WORKING_HOURS_START.minute)
        ) / 60
        if isinstance(start, str) and isinstance(end, str):
            current_datetime = datetime.strptime(start, constants.DT_FORMAT)
            delivery_datetime = datetime.strptime(end, constants.DT_FORMAT)
        else:
            current_datetime = start
            delivery_datetime = end
        if current_datetime.date() == delivery_datetime.date():
            if delivery_datetime >= delivery_datetime.replace(
                hour=self.WORKING_HOURS_START.hour,
                minute=self.WORKING_HOURS_START.minute,
            ):
                if not is_work_break_time:
                    total_time_lunch_break = self.compute_lunch_break(
                        current_datetime, delivery_datetime
                    )
                else:
                    total_time_lunch_break = 0
                return (
                    min(
                        delivery_datetime.replace(
                            hour=self.WORKING_HOURS_END.hour,
                            minute=self.WORKING_HOURS_END.minute,
                        ),
                        delivery_datetime,
                    )
                    - max(
                        current_datetime,
                        current_datetime.replace(
                            hour=self.WORKING_HOURS_START.hour,
                            minute=self.WORKING_HOURS_START.minute,
                        ),
                    )
                ).total_seconds() / 3600 - total_time_lunch_break
            else:
                return 0

        # Create an empty list to store the days
        all_days = []
        keep_current = current_datetime
        # Iterate through the days
        while current_datetime.date() <= delivery_datetime.date():
            all_days.append(current_datetime)
            current_datetime += timedelta(days=1)

        # Initialize variable to track remaining hours
        total_hours = 0

        # Process for couple middle dates
        all_days = all_days[1:-1]
        for d in all_days:
            if not self.is_day_off(d):
                total_hours += hours_working_in_day
        # Process current_date
        if not self.is_day_off(keep_current):
            t = (
                keep_current.replace(
                    hour=self.WORKING_HOURS_END.hour,
                    minute=self.WORKING_HOURS_END.minute,
                )
                - max(
                    keep_current,
                    keep_current.replace(
                        hour=self.WORKING_HOURS_START.hour,
                        minute=self.WORKING_HOURS_START.minute,
                    ),
                )
            ).total_seconds() / 3600
        else:
            t = 0
        if t > 0:
            total_hours += t

        if not self.is_day_off(delivery_datetime):
            t = (
                min(
                    delivery_datetime.replace(
                        hour=self.WORKING_HOURS_END.hour,
                        minute=self.WORKING_HOURS_END.minute,
                    ),
                    delivery_datetime,
                )
                - min(
                    delivery_datetime,
                    delivery_datetime.replace(
                        hour=self.WORKING_HOURS_START.hour,
                        minute=self.WORKING_HOURS_START.minute,
                    ),
                )
            ).total_seconds() / 3600
        else:
            t = 0
        if t > 0:
            total_hours += t
        total_time_lunch_break = self.compute_lunch_break(
            keep_current, delivery_datetime
        )
        total_hours = total_hours - total_time_lunch_break
        if total_hours < 0:
            print(start, end)
            raise RuntimeError("Start time has to be less than end time")
        return total_hours

    def get_available_time(
        self, start_time, end_time, schedule_plan, type
    ) -> List[Tuple]:
        # Initialize a list to store the available time periods
        available_periods = []
        allocated_periods = [
            (x[0], x[1]) if x[1] <= end_time else (x[0], end_time)
            for x in schedule_plan
            if x[0] < end_time
        ]
        if len(allocated_periods) > 0:
            temp_allocated_periods = allocated_periods.copy()
            # Initialize the current time as the start time
            current_time = start_time
            a_p_0 = temp_allocated_periods[0]
            temp_allocated_periods.pop(0)
            if current_time < a_p_0[0]:
                available_periods.append((current_time, a_p_0[0]))
            current_time = a_p_0[1]
            for allocated_period in temp_allocated_periods:
                available_periods.append((current_time, allocated_period[0]))
                current_time = allocated_period[1]
            if current_time < end_time:
                available_periods.append((current_time, end_time))
        else:
            available_periods = [(start_time, end_time)]
        if type == "machine":
            return available_periods

        new_available_periods = []
        for p in available_periods:
            if p[0] == p[1]:
                continue
            if p[0].date() == p[1].date():
                if (
                    p[1].time() <= self.WORKING_HOURS_START
                    or p[0].time() >= self.WORKING_HOURS_END
                ):
                    continue
                if (
                    p[1].time() <= self.WORKING_HOURS_END
                    and p[0].time() >= self.WORKING_HOURS_START
                ):
                    new_available_periods.append(p)
                else:
                    new_available_periods.append(
                        (
                            max(
                                p[0],
                                p[0].replace(
                                    hour=self.WORKING_HOURS_START.hour,
                                    minute=self.WORKING_HOURS_START.minute,
                                ),
                            ),
                            min(
                                p[1],
                                p[1].replace(
                                    hour=self.WORKING_HOURS_END.hour,
                                    minute=self.WORKING_HOURS_END.minute,
                                ),
                            ),
                        )
                    )
                continue
            diff_days = (p[1] - p[0]).days
            if (
                p[0].replace(
                    hour=self.WORKING_HOURS_END.hour,
                    minute=self.WORKING_HOURS_END.minute,
                )
                != p[0]
                and p[0].time() < self.WORKING_HOURS_END
            ):
                new_available_periods.append(
                    (
                        max(
                            p[0],
                            p[0].replace(
                                hour=self.WORKING_HOURS_START.hour,
                                minute=self.WORKING_HOURS_START.minute,
                            ),
                        ),
                        p[0].replace(
                            hour=self.WORKING_HOURS_END.hour,
                            minute=self.WORKING_HOURS_END.minute,
                        ),
                    )
                )
            if diff_days > 1:
                dt_current_date_temp = p[0].replace(minute=0)
                while dt_current_date_temp.date() < p[1].date():
                    dt_current_date_temp += timedelta(days=1)
                    if self.is_day_off(dt_current_date_temp):
                        continue
                    temp_start = dt_current_date_temp.replace(
                        hour=self.WORKING_HOURS_START.hour,
                        minute=self.WORKING_HOURS_START.minute,
                    )
                    if dt_current_date_temp.date() == p[1].date():
                        if temp_start == p[1]:
                            break
                        if (
                            p[1].replace(
                                hour=self.WORKING_HOURS_START.hour,
                                minute=self.WORKING_HOURS_START.minute,
                            )
                            >= p[1]
                        ):
                            temp_end = p[1] - timedelta(days=1)
                            temp_end = temp_end.replace(
                                hour=self.WORKING_HOURS_END.hour,
                                minute=self.WORKING_HOURS_END.minute,
                            )
                        elif p[1].time() > self.WORKING_HOURS_END:
                            temp_end = p[1].replace(
                                hour=self.WORKING_HOURS_END.hour,
                                minute=self.WORKING_HOURS_END.minute,
                            )
                        else:
                            temp_end = p[1]
                    else:
                        temp_end = dt_current_date_temp.replace(
                            hour=self.WORKING_HOURS_END.hour,
                            minute=self.WORKING_HOURS_END.minute,
                        )
                    new_available_periods.append((temp_start, temp_end))
            else:
                if (p[1].date() - p[0].date()).days == 2:
                    new_available_periods.append(
                        (
                            p[0].replace(
                                hour=self.WORKING_HOURS_START.hour,
                                minute=self.WORKING_HOURS_START.minute,
                            )
                            + timedelta(days=1),
                            p[0].replace(
                                hour=self.WORKING_HOURS_END.hour,
                                minute=self.WORKING_HOURS_END.minute,
                            )
                            + timedelta(days=1),
                        )
                    )

                if (
                    p[1].time() != self.WORKING_HOURS_START
                    and p[1].time() > self.WORKING_HOURS_START
                ):
                    new_available_periods.append(
                        (
                            p[1].replace(
                                hour=self.WORKING_HOURS_START.hour,
                                minute=self.WORKING_HOURS_START.minute,
                            ),
                            min(
                                p[1],
                                p[1].replace(
                                    hour=self.WORKING_HOURS_END.hour,
                                    minute=self.WORKING_HOURS_END.minute,
                                ),
                            ),
                        )
                    )
        new_available_periods = [
            x for x in new_available_periods if not self.is_day_off(x[0])
        ]
        return new_available_periods

    def time_analyzer(
        self, dict_worker_avail_time, total_hours, is_work_break_time=False
    ) -> Dict[str, Dict]:
        dict_time_analyzer = {}

        for worker, avail_time in dict_worker_avail_time.items():
            total_remaining_hours = 0
            clean_avail_time = []
            for start_time, end_time in avail_time:
                total_remaining_hours += self.get_diff_hours(
                    start_time, end_time, is_work_break_time
                )
                if total_remaining_hours != 0:
                    clean_avail_time.append((start_time, end_time))
            try:
                operating_rate = (total_hours - total_remaining_hours) / total_hours
            except ZeroDivisionError:
                operating_rate = 0
            dict_time_analyzer[worker] = {
                "free_time": clean_avail_time,
                "total_available_hours": total_remaining_hours,
                "operating_rate": operating_rate,
            }
        return dict_time_analyzer

    def avail_time_grouper(
        self, avail_time_list, is_work_break_time=False
    ) -> Dict[str, List[List]]:
        group_avail_time = {}

        for id, ts_list in avail_time_list.items():
            group_avail_time[id] = []
            if not len(ts_list):
                continue
            temp = [ts_list[0]]
            i = 0
            while i < len(ts_list) - 1:
                if ts_list[i][1] == ts_list[i + 1][0]:
                    temp[-1] = (temp[-1][0], ts_list[i + 1][-1])
                elif (
                    (ts_list[i][1].time(), ts_list[i + 1][0].time())
                    in self.list_breaktime
                    and ts_list[i][1].date() == ts_list[i + 1][0].date()
                    and not is_work_break_time
                ):
                    temp.append(ts_list[i + 1])
                elif (
                    ts_list[i][1].time() == self.WORKING_HOURS_END
                    and ts_list[i + 1][0].time() == self.WORKING_HOURS_START
                ):
                    if self.is_all_day_off(ts_list[i][1], ts_list[i + 1][0]):
                        temp.append(ts_list[i + 1])
                    else:
                        if temp:
                            group_avail_time[id].append(temp)
                        temp = [ts_list[i + 1]]
                else:
                    if temp:
                        group_avail_time[id].append(temp)
                    temp = [ts_list[i + 1]]
                i += 1

            if temp:
                group_avail_time[id].append(temp)
        return group_avail_time

    def get_overtime_hours(self, start, end, is_work_break_time=False) -> float:
        with self.adjust_schedule_time():
            working_not_overtime = self.get_diff_hours(
                start, end, is_work_break_time=is_work_break_time
            )
        with self.adjust_schedule_time(overtime=True):
            overtime = (
                self.get_diff_hours(start, end, is_work_break_time=is_work_break_time)
                - working_not_overtime
            )
        return overtime

    def rework_worker_avail_time_with_breaktime(
        self, worker_avail_time, start_break, end_break
    ):
        new_worker_avail_time = {}
        ### process time lunch break
        for worker, time_avail in worker_avail_time.items():
            new_worker_avail_time[worker] = []
            for time in time_avail:
                start_time = time[0]
                start_break = time[0].replace(
                    hour=start_break.hour, minute=start_break.minute
                )
                end_time = time[1]
                end_break = time[1].replace(
                    hour=end_break.hour, minute=end_break.minute
                )
                if start_time >= end_break or end_time <= start_break:
                    new_worker_avail_time[worker].extend([(start_time, end_time)])
                    continue
                intervals = []
                if start_time < start_break:
                    intervals.append((start_time, min(end_time, start_break)))
                if end_time > end_break:
                    intervals.append((max(start_time, end_break), end_time))
                new_worker_avail_time[worker].extend(intervals)
        return new_worker_avail_time

    def get_operating_rate(self, rs_id, total_allocated_time, type):
        try:
            total_time_worked_worker = self.dict_total_worked_time[type].get(rs_id, 0)
            total_time_worked_worker = (
                total_time_worked_worker / 60 if total_time_worked_worker else 0
            )
            operating_rate_worker = total_time_worked_worker / total_allocated_time
            operating_rate_worker = (
                operating_rate_worker if operating_rate_worker <= 1 else 1
            )
            return operating_rate_worker
        except ZeroDivisionError:
            return 0

    def shift_end_to_next_holiday(self, end_process, is_work_break_time):
        while True:
            if self.is_day_off(end_process):
                end_process = end_process + timedelta(days=1)
                end_process = datetime.combine(
                    end_process.date(), self.WORKING_HOURS_START
                )
            else:
                start_of_working_hours = datetime.combine(
                    end_process.date(), self.WORKING_HOURS_START
                )
                end_of_working_hours = datetime.combine(
                    end_process.date(), self.WORKING_HOURS_END
                )
                if not is_work_break_time:
                    if (
                        datetime.combine(end_process.date(), self.MORNING_END_TIME)
                        <= end_process
                        <= datetime.combine(
                            end_process.date(), self.AFTERNOON_START_TIME
                        )
                    ):
                        end_process = datetime.combine(
                            end_process.date(), self.AFTERNOON_START_TIME
                        )
                if not (start_of_working_hours <= end_process < end_of_working_hours):
                    if end_process <= start_of_working_hours:
                        end_process = start_of_working_hours
                    elif end_process >= end_of_working_hours:
                        end_process = start_of_working_hours + timedelta(days=1)
                if not self.is_day_off(end_process):
                    break
        return end_process

    def generate_working_intervals(
        self,
        start_dt: datetime,
        min_total_minutes: int,
        is_work_break_time: bool,
        deadline: Optional[datetime] = None,
    ) -> Tuple[List[Tuple[datetime, datetime]], bool]:
        if not min_total_minutes:
            return [(start_dt, start_dt)], False
        if deadline and start_dt >= deadline:
            return [(start_dt, start_dt)], True
        result = []
        current_date = start_dt
        total_minutes = 0
        while True:
            if self.is_day_off(current_date):
                current_date += timedelta(days=1)
                continue
            for work_start, work_end in self.WORKING_HOUR_RANGE:
                intervals = []
                previous = work_start
                if not is_work_break_time:
                    for b_start, b_end in self.list_breaktime:
                        if b_start >= previous and b_start < work_end:
                            intervals.append((previous, min(b_start, work_end)))
                            previous = max(b_end, previous)
                if previous < work_end:
                    intervals.append((previous, work_end))

                for s_time, e_time in intervals:
                    s_dt = datetime.combine(current_date.date(), s_time)
                    e_dt = datetime.combine(current_date.date(), e_time)

                    if e_dt <= start_dt:
                        continue
                    if deadline and deadline <= s_dt:
                        return result, True
                    s_dt = max(s_dt, start_dt)
                    duration_minutes = int((e_dt - s_dt).total_seconds() / 60)

                    if total_minutes + duration_minutes >= min_total_minutes:
                        needed = min_total_minutes - total_minutes
                        final_end = s_dt + timedelta(minutes=needed)
                        if deadline and s_dt <= deadline <= final_end:
                            result.append((s_dt, deadline))
                            return result, True
                        else:
                            result.append((s_dt, final_end))
                        return result, False
                    if deadline and s_dt <= deadline <= s_dt:
                        result.append((s_dt, deadline))
                        return result, True
                    else:
                        result.append((s_dt, e_dt))
                    total_minutes += duration_minutes

            current_date += timedelta(days=1)

    def nightshift_timegrouper_convert(self, time_grouper):
        time_grouper_tmp = {}
        for key, groups in time_grouper.items():
            for group in groups:
                time_grouper_tmp.setdefault(key, []).append(
                    [(group[0][0], group[-1][1])]
                )
        time_grouper = time_grouper_tmp
        return time_grouper

    def resource_filter(
        self, process_unit, time_grouper, time_grouper_machine, deadline=None
    ):
        schedule_predict_step_dict = defaultdict(lambda: defaultdict(list))
        schedule_predict_saver_dict = defaultdict(dict)
        list_valid_ids = []
        list_final_ids = []
        step = constants.PRE_PROCESS_STEP
        p_time = process_unit["timing_details"][step]
        list_final_ids_tmp = []
        for id, ts_list in time_grouper.items():
            worker_id, machine_id = map(lambda x: int(float(x)), id.split("___"))
            is_auto = self.factory_rs_info["dict_machine_auto"].get(machine_id, False)
            for i, grouped_time in enumerate(ts_list):
                allocated_time_list_tmp, exceed_end_max = (
                    self.generate_working_intervals(
                        start_dt=grouped_time[0][0],
                        min_total_minutes=p_time,
                        is_work_break_time=process_unit["is_work_break_time"],
                        deadline=deadline,
                    )
                )
                if exceed_end_max:
                    list_final_ids_tmp.append(f"{id}___{i}")
                schedule_predict_step_dict[step][f"{id}___{i}"].append(
                    allocated_time_list_tmp
                )
                schedule_predict_saver_dict[f"{id}___{i}"][
                    step
                ] = allocated_time_list_tmp
        _, per_group_verdict = self.validate_queries_inside_availability(
            time_grouper_qr=schedule_predict_step_dict[step],
            time_grouper=time_grouper,
            deadline=deadline,
            auto=False,
        )
        per_group_verdict = per_group_verdict.filter(
            pl.col("all_queries_contained")
        ).with_columns(
            (pl.col("id") + "___" + pl.col("group").cast(pl.Utf8)).alias("id_group")
        )
        list_valid_ids.extend(per_group_verdict["id_group"].to_list())
        list_final_ids.extend(list(set(list_valid_ids) & set(list_final_ids_tmp)))
        list_valid_ids = list(set(list_valid_ids) - set(list_final_ids))
        for step, p_time in process_unit["timing_details"].items():
            if step == constants.PRE_PROCESS_STEP:
                continue
            list_final_ids_tmp = []
            for id in list_valid_ids:
                _, machine_id, _ = map(lambda x: int(float(x)), id.split("___"))
                is_auto = self.factory_rs_info["dict_machine_auto"].get(
                    machine_id, False
                )
                current_end = schedule_predict_saver_dict[id][
                    constants.PREVIOUS_PROCESS_STEP[step]
                ][-1][-1]
                if (
                    process_unit["is_nightshift_expected"]
                    and step == constants.PROCESSING_STEP
                ):
                    if not process_unit["is_work_break_time"]:
                        allocated_time_list_tmp, exceed_end_max = (
                            self.allocate_processing_time_with_breaks(
                                start=current_end,
                                processing_minutes=p_time,
                                deadline=deadline,
                            )
                        )
                    else:
                        exceed_end_max = False
                        end_mid_process = current_end + timedelta(minutes=p_time)
                        if deadline and current_end <= deadline <= end_mid_process:
                            allocated_time_list_tmp = [(current_end, deadline)]
                            exceed_end_max = True
                        else:
                            allocated_time_list_tmp = [(current_end, end_mid_process)]
                    if process_unit["timing_details"][constants.POST_PROCESS_STEP] > 0:
                        end_mid_process = self.shift_end_to_next_holiday(
                            allocated_time_list_tmp[-1][-1],
                            process_unit["is_work_break_time"],
                        )
                        if (
                            deadline
                            and allocated_time_list_tmp[-1][0]
                            <= deadline
                            <= end_mid_process
                        ):
                            allocated_time_list_tmp[-1] = (
                                allocated_time_list_tmp[-1][0],
                                deadline,
                            )
                            exceed_end_max = True
                        else:
                            allocated_time_list_tmp[-1] = (
                                allocated_time_list_tmp[-1][0],
                                end_mid_process,
                            )
                else:
                    allocated_time_list_tmp, exceed_end_max = (
                        self.generate_working_intervals(
                            start_dt=current_end,
                            min_total_minutes=p_time,
                            is_work_break_time=process_unit["is_work_break_time"],
                            deadline=deadline,
                        )
                    )
                if exceed_end_max:
                    list_final_ids_tmp.append(id)
                step_auto = (
                    "processing_auto"
                    if is_auto and step == constants.PROCESSING_STEP
                    else step
                )
                schedule_predict_step_dict[step_auto][f"{id}"].append(
                    allocated_time_list_tmp
                )
                schedule_predict_saver_dict[f"{id}"][step] = allocated_time_list_tmp
            list_valid_ids = []
            if (
                step == constants.PROCESSING_STEP
                and schedule_predict_step_dict["processing_auto"]
            ):
                time_grouper_machine = self.avail_time_grouper(time_grouper_machine)
                time_grouper_local = time_grouper_machine
                if process_unit["is_nightshift_expected"]:
                    time_grouper_local = self.nightshift_timegrouper_convert(
                        time_grouper_machine
                    )
                _, per_group_verdict = self.validate_queries_inside_availability(
                    time_grouper_qr=schedule_predict_step_dict["processing_auto"],
                    time_grouper=time_grouper_local,
                    deadline=deadline,
                    auto=True,
                )
                per_group_verdict = per_group_verdict.filter(
                    pl.col("all_queries_contained")
                ).with_columns(
                    (pl.col("id") + "___" + pl.col("group").cast(pl.Utf8)).alias(
                        "id_group"
                    )
                )
                list_valid_ids.extend(per_group_verdict["id_group"].to_list())
            if schedule_predict_step_dict[step]:
                time_grouper_local = time_grouper
                if process_unit["is_nightshift_expected"]:
                    time_grouper_local = self.nightshift_timegrouper_convert(
                        time_grouper
                    )
                _, per_group_verdict = self.validate_queries_inside_availability(
                    time_grouper_qr=schedule_predict_step_dict[step],
                    time_grouper=time_grouper_local,
                    deadline=deadline,
                    auto=False,
                )
                per_group_verdict = per_group_verdict.filter(
                    pl.col("all_queries_contained")
                ).with_columns(
                    (pl.col("id") + "___" + pl.col("group").cast(pl.Utf8)).alias(
                        "id_group"
                    )
                )
                list_valid_ids.extend(per_group_verdict["id_group"].to_list())
            list_final_ids.extend(list(set(list_valid_ids) & set(list_final_ids_tmp)))
            list_valid_ids = list(set(list_valid_ids) - set(list_final_ids_tmp))
        return list_valid_ids + list_final_ids, schedule_predict_step_dict

    def build_allocated_time_list(
        self,
        process_unit,
        time_grouper,
        time_grouper_machine,
        total_allocated_time,
        overtime,
        deadline=None,
    ):
        decision_dict = {}
        best_end_process = None
        list_valid_ids, schedule_predict_step_dict = self.resource_filter(
            process_unit=process_unit,
            time_grouper=time_grouper,
            time_grouper_machine=time_grouper_machine,
            deadline=deadline,
        )
        dict_operating_rate_worker = {}
        dict_operating_rate_machine = {}
        dict_start_of_ids = {}
        for id in list_valid_ids:
            dict_start_of_ids[id] = schedule_predict_step_dict[
                constants.PRE_PROCESS_STEP
            ][id][0][0][0]
        for id, start_process in sorted(dict_start_of_ids.items(), key=lambda x: x[1]):
            allocated_time_dict = defaultdict()
            current_end = start_process
            for step, p_time in process_unit["timing_details"].items():
                if (
                    process_unit["is_nightshift_expected"]
                    and step == constants.PROCESSING_STEP
                ):
                    if not process_unit["is_work_break_time"]:
                        allocated_time_list_tmp, _ = (
                            self.allocate_processing_time_with_breaks(
                                start=current_end,
                                processing_minutes=p_time,
                                deadline=deadline,
                            )
                        )
                    else:
                        end_mid_process = current_end + timedelta(minutes=p_time)
                        if deadline and current_end <= deadline <= end_mid_process:
                            allocated_time_list_tmp = [
                                (current_end, deadline),
                                (deadline, end_mid_process),
                            ]
                        else:
                            allocated_time_list_tmp = [(current_end, end_mid_process)]
                    if process_unit["timing_details"][constants.POST_PROCESS_STEP] > 0:
                        end_mid_process = self.shift_end_to_next_holiday(
                            allocated_time_list_tmp[-1][-1],
                            process_unit["is_work_break_time"],
                        )
                        if (
                            deadline
                            and allocated_time_list_tmp[-1][0]
                            <= deadline
                            <= end_mid_process
                        ):
                            allocated_time_list_tmp[-1] = (
                                allocated_time_list_tmp[-1][0],
                                deadline,
                            )
                            allocated_time_list_tmp.append((deadline, end_mid_process))
                        else:
                            allocated_time_list_tmp[-1] = (
                                allocated_time_list_tmp[-1][0],
                                end_mid_process,
                            )
                else:
                    allocated_time_list_tmp, _ = self.generate_working_intervals(
                        start_dt=current_end,
                        min_total_minutes=p_time,
                        is_work_break_time=process_unit["is_work_break_time"],
                    )
                allocated_time_dict[step] = allocated_time_list_tmp
                current_end = allocated_time_list_tmp[-1][-1]

            end_process = self.max_end_date_in_dict(allocated_time_dict)
            if best_end_process is not None and end_process > best_end_process:
                continue
            worker_id, machine_id, _ = map(lambda x: int(float(x)), id.split("___"))
            is_auto = self.factory_rs_info["dict_machine_auto"].get(machine_id, False)
            operating_rate_worker = dict_operating_rate_worker.setdefault(
                worker_id,
                self.get_operating_rate(
                    rs_id=worker_id,
                    total_allocated_time=total_allocated_time,
                    type="worker",
                ),
            )
            operating_rate_machine = dict_operating_rate_machine.setdefault(
                machine_id,
                self.get_operating_rate(
                    rs_id=machine_id,
                    total_allocated_time=total_allocated_time,
                    type="machine",
                ),
            )
            decision_dict[f"{id}"] = {
                "end": self.max_end_date_in_dict(allocated_time_dict),
                "overtime_hours": (
                    self.get_overtime_hours(
                        start_process, end_process, process_unit["is_work_break_time"]
                    )
                    if overtime
                    else 0
                ),
                "operating_rate": operating_rate_worker + operating_rate_machine,
                "is_not_auto": not is_auto,
                "allocated_time_dict": allocated_time_dict,
            }
            best_end_process = end_process
        enough_time_list = sorted(
            decision_dict.items(),
            key=lambda x: (
                x[1]["overtime_hours"],
                x[1]["end"],
                x[1]["is_not_auto"],
                x[1]["operating_rate"],
            ),
        )
        allocated_time_dict = []
        worker_id = ""
        machine_id = ""
        if len(enough_time_list):
            selected_resource = enough_time_list[0]
            (
                worker_id,
                machine_id,
                _,
            ) = selected_resource[
                0
            ].split("___")
            allocated_time_dict = selected_resource[1]["allocated_time_dict"]
        return worker_id, machine_id, allocated_time_dict

    def build_enough_time_list_unmanned_process_nightshift_priority(
        self,
        process_unit,
        time_grouper,
        time_grouper_machine,
        total_allocated_time,
        overtime,
    ):
        decision_dict = {}
        dict_operating_rate_worker = {}
        dict_operating_rate_machine = {}
        list_valid_ids, _ = self.resource_filter(
            process_unit, time_grouper, time_grouper_machine
        )
        for id, ts_list in time_grouper.items():
            worker_id = int(float(id.split("___")[0]))
            process_id = int(float(id.split("___")[1]))
            ts_list_machine = time_grouper_machine[process_id]
            allocated_time_dict = defaultdict(list)
            operating_rate_worker = dict_operating_rate_worker.setdefault(
                worker_id,
                self.get_operating_rate(
                    rs_id=worker_id,
                    total_allocated_time=total_allocated_time,
                    type="worker",
                ),
            )
            operating_rate_machine = dict_operating_rate_machine.setdefault(
                process_id,
                self.get_operating_rate(
                    rs_id=process_id,
                    total_allocated_time=total_allocated_time,
                    type="machine",
                ),
            )
            for index in range(len(ts_list)):
                if f"{id}___{index}" not in list_valid_ids:
                    continue
                previous_processing_time = process_unit["timing_details"][
                    constants.PRE_PROCESS_STEP
                ]
                mid_processing_time = process_unit["timing_details"][
                    constants.PROCESSING_STEP
                ]
                post_processing_time = process_unit["timing_details"][
                    constants.POST_PROCESS_STEP
                ]
                grouped_time = ts_list[index]
                if not len(grouped_time):
                    continue
                grouped_time_df = pd.DataFrame(
                    data=grouped_time, columns=["start", "end"]
                )
                grouped_time_df["diff"] = (
                    grouped_time_df["end"] - grouped_time_df["start"]
                )
                grouped_time_df["cumsum"] = grouped_time_df["diff"].cumsum()

                grouped_time_df = grouped_time_df[
                    (
                        grouped_time_df["cumsum"]
                        >= timedelta(minutes=previous_processing_time)
                    )
                    & (grouped_time_df["end"].dt.hour >= self.WORKING_HOURS_END.hour)
                ]
                if grouped_time_df.empty:
                    continue
                available_pre_process = grouped_time[: (grouped_time_df.index[0] + 1)]
                if previous_processing_time > 0:
                    while previous_processing_time > 0 and available_pre_process:
                        start_time, end_time = available_pre_process[-1]
                        time_difference = (end_time - start_time).total_seconds() / 60
                        if time_difference >= previous_processing_time:
                            start_pre_process = end_time - timedelta(
                                minutes=previous_processing_time
                            )
                            previous_processing_time = 0
                            allocated_time_dict[constants.PRE_PROCESS_STEP].append(
                                (start_pre_process, end_time)
                            )
                            if start_pre_process == start_time:
                                available_pre_process.pop(-1)
                        else:
                            allocated_time_dict[constants.PRE_PROCESS_STEP].append(
                                available_pre_process.pop(-1)
                            )
                            previous_processing_time -= time_difference
                else:
                    allocated_time_dict[constants.PRE_PROCESS_STEP] = [
                        (available_pre_process[-1][-1], available_pre_process[-1][-1])
                    ]
                allocated_time_dict[constants.PRE_PROCESS_STEP] = list(
                    reversed(allocated_time_dict[constants.PRE_PROCESS_STEP])
                )

                is_mid_process_not_suitable = False
                if mid_processing_time > 0:
                    end_mid_process = allocated_time_dict[constants.PRE_PROCESS_STEP][
                        -1
                    ][-1] + timedelta(minutes=mid_processing_time)
                    allocated_time_dict[constants.PROCESSING_STEP] = [
                        (
                            allocated_time_dict[constants.PRE_PROCESS_STEP][-1][-1],
                            end_mid_process,
                        )
                    ]
                else:
                    allocated_time_dict[constants.PROCESSING_STEP] = [
                        (
                            allocated_time_dict[constants.PRE_PROCESS_STEP][-1][-1],
                            allocated_time_dict[constants.PRE_PROCESS_STEP][-1][-1],
                        )
                    ]
                start_mid_process = allocated_time_dict[constants.PROCESSING_STEP][0][0]
                end_mid_process = allocated_time_dict[constants.PROCESSING_STEP][-1][-1]
                for gr_time_machine in ts_list_machine:
                    s_machine = gr_time_machine[0]
                    e_machine = gr_time_machine[1]
                    is_mid_process_not_suitable = True
                    if s_machine <= start_mid_process <= end_mid_process <= e_machine:
                        is_mid_process_not_suitable = False
                        break
                if is_mid_process_not_suitable:
                    continue
                end_mid_process = self.shift_end_to_next_holiday(
                    end_mid_process, process_unit["is_work_break_time"]
                )
                allocated_time_dict[constants.PROCESSING_STEP][-1] = (
                    allocated_time_dict[constants.PROCESSING_STEP][-1][0],
                    end_mid_process,
                )
                time_ranges = ts_list[index]
                is_post_process_not_suitable = False
                if not post_processing_time == 0:
                    while end_mid_process >= time_ranges[-1][-1]:
                        index += 1
                        if index > len(ts_list) - 1:
                            is_post_process_not_suitable = True
                            break
                        time_ranges = ts_list[index]
                    while time_ranges:
                        start_time, end_time = time_ranges[0]
                        if end_time <= end_mid_process:
                            time_ranges.pop(0)
                        else:
                            time_ranges[0] = (end_mid_process, end_time)
                            break
                    else:
                        is_post_process_not_suitable = True
                    if is_post_process_not_suitable:
                        continue
                    is_post_process_not_suitable = False
                    if post_processing_time > 0:
                        while post_processing_time > 0 and time_ranges:
                            start_time, end_time = time_ranges[0]
                            time_difference = end_time - start_time

                            if (
                                time_difference.total_seconds()
                                >= post_processing_time * 60
                            ):
                                end_post_process = start_time + timedelta(
                                    minutes=post_processing_time
                                )
                                post_processing_time = 0
                                allocated_time_dict[constants.POST_PROCESS_STEP].append(
                                    (start_time, end_post_process)
                                )
                                if end_post_process == end_time:
                                    time_ranges.pop(0)
                                else:
                                    time_ranges[0] = (end_post_process, end_time)
                            else:
                                if len(time_ranges):
                                    allocated_time_dict[
                                        constants.POST_PROCESS_STEP
                                    ].append(time_ranges.pop(0))
                                    post_processing_time -= (
                                        time_difference.total_seconds() // 60
                                    )
                                    if not len(time_ranges) and post_processing_time:
                                        is_post_process_not_suitable = True
                                        break
                                else:
                                    is_post_process_not_suitable = True
                                    break
                    else:
                        allocated_time_dict[constants.POST_PROCESS_STEP].append(
                            (time_ranges[0][0], time_ranges[0][0])
                        )

                    if is_post_process_not_suitable:
                        continue
                else:
                    if end_mid_process > time_ranges[-1][-1]:
                        continue
                    allocated_time_dict[constants.POST_PROCESS_STEP].append(
                        (end_mid_process, end_mid_process)
                    )
                end_process = allocated_time_dict[constants.POST_PROCESS_STEP][-1][-1]
                # build decision dict
                decision_dict[f"{id}___{index}"] = {
                    "end": end_process,
                    "overtime_hours": (
                        self.get_overtime_hours(
                            allocated_time_dict[constants.PRE_PROCESS_STEP][0][0],
                            end_process,
                        )
                        if overtime
                        else 0
                    ),
                    "operating_rate": operating_rate_worker + operating_rate_machine,
                    "allocated_time_dict": allocated_time_dict,
                    "is_not_auto": not self.factory_rs_info["dict_machine_auto"].get(
                        process_id, False
                    ),
                    "night_time": self.compute_night_time_hours(
                        allocated_time_dict[constants.PROCESSING_STEP]
                    ),
                }

        enough_time_list = sorted(
            decision_dict.items(),
            key=lambda x: (
                x[1]["end"],
                x[1]["overtime_hours"],
                x[1]["operating_rate"],
                -x[1]["night_time"],
            ),
        )
        worker_id = ""
        process_id = ""
        list_of_allocated_time_dict = []
        if len(enough_time_list):
            selected_resource = enough_time_list[0]
            (
                worker_id,
                process_id,
                _,
            ) = selected_resource[
                0
            ].split("___")
            list_of_allocated_time_dict.append(
                (worker_id, process_id, selected_resource[1]["allocated_time_dict"])
            )
        return list_of_allocated_time_dict

    def _time_grouper_to_df(
        self, time_grouper: Dict[str, List[List[Tuple[datetime, datetime]]]]
    ) -> pl.DataFrame:
        rows = []
        for _id, group_lists in time_grouper.items():
            for g_idx, tuple_list in enumerate(group_lists):
                for s, e in tuple_list:
                    rows.append({"id": _id, "group": g_idx, "start": s, "end": e})
        return (
            pl.from_dicts(rows)
            .with_columns(
                pl.col("start").cast(pl.Datetime("us")),
                pl.col("end").cast(pl.Datetime("us")),
            )
            .sort(["id", "group", "start"])
        )

    def validate_queries_inside_availability(
        self, time_grouper_qr, time_grouper, deadline=None, auto=False
    ):
        avail_df = self._time_grouper_to_df(time_grouper)
        qr_df = self._time_grouper_to_df(time_grouper_qr)
        qr_df = (
            qr_df.with_columns(qr_df["id"].str.split_exact("___", 2).alias("id_parts"))
            .with_columns(
                [
                    pl.col("id_parts").struct.field("field_0").alias("part1"),
                    pl.col("id_parts").struct.field("field_1").alias("part2"),
                    pl.col("id_parts").struct.field("field_2").alias("part3"),
                ]
            )
            .drop("id_parts")
        )
        id_name = "id"
        if auto:
            id_name = "id_machine"
            qr_df = (
                qr_df.with_columns(
                    qr_df["id"].str.split_exact("___", 2).alias("id_parts")
                )
                .with_columns(
                    [
                        pl.col("id_parts")
                        .struct.field("field_1")
                        .cast(pl.Int64)
                        .alias(id_name),
                    ]
                )
                .drop("id_parts")
            )
            avail_df = avail_df.with_columns(pl.col("id").alias(id_name))
        qr_df = qr_df.with_columns(
            [
                (pl.col("part1") + "___" + pl.col("part2")).alias("id"),
                pl.col("part3").cast(pl.Int64).alias("group"),
            ]
        ).drop(["part1", "part2", "part3"])
        joined = qr_df.join(
            avail_df.rename(
                {"start": "start_avail", "end": "end_avail", "group": "group_avail"}
            ),
            on=[id_name],
            how="inner",
        )
        if deadline is not None:
            joined = joined.with_columns(pl.lit(deadline).alias("deadline"))
            check = joined.with_columns(
                (
                    (
                        (pl.col("start_avail") <= pl.col("start"))
                        & (pl.col("end_avail") >= pl.col("end"))
                    )
                    | (pl.col("start") >= pl.col("deadline"))
                    | (pl.col("start") == pl.col("end"))
                ).alias("contained")
            )
        else:
            check = joined.with_columns(
                (
                    (pl.col("start_avail") <= pl.col("start"))
                    & (pl.col("end_avail") >= pl.col("end"))
                ).alias("contained")
            )
        per_interval = (
            check.group_by(["id", "group", "start", "end"])
            .agg(pl.col("contained").any().alias("is_contained"))
            .sort(["id", "group", "start"])
        )
        per_group = (
            per_interval.group_by(["id", "group"])
            .agg(pl.col("is_contained").all().alias("all_queries_contained"))
            .sort(["id", "group"])
        )
        covered_debug = (
            check.filter(pl.col("contained"))
            .group_by(["id", "group", "start", "end"])
            .agg(
                pl.col("start_avail").first().alias("cover_start_avail"),
                pl.col("end_avail").first().alias("cover_end_avail"),
            )
            .sort(["id", "group", "start"])
        )
        per_interval_with_debug = per_interval.join(
            covered_debug, on=["id", "group", "start", "end"], how="left"
        )
        return per_interval_with_debug, per_group

    def build_allocated_time_list_external(
        self,
        start,
        total_processing_time,
        is_included_holiday=False,
        worker_id=False,
        process_id=False,
    ):
        start_of_day = start.replace(
            hour=self.WORKING_HOURS_START.hour, minute=self.WORKING_HOURS_START.minute
        )
        end_of_day = start.replace(
            hour=self.WORKING_HOURS_END.hour, minute=self.WORKING_HOURS_END.minute
        )
        if start < start_of_day:
            start = start_of_day
        elif start >= end_of_day:
            start = start_of_day + timedelta(days=1)
        if is_included_holiday and worker_id:
            while self.is_day_off_external(day=start, organizations=worker_id):
                start = (start + timedelta(days=1)).replace(
                    hour=self.WORKING_HOURS_START.hour,
                    minute=self.WORKING_HOURS_START.minute,
                )
        end_process = start + timedelta(minutes=total_processing_time)
        allocated_time_list = [
            (start, start),
            (start, end_process),
            (end_process, end_process),
        ]
        return str(worker_id), str(process_id), allocated_time_list

    def allocate_processing_time_with_breaks(
        self, start: datetime, processing_minutes: int, deadline=None
    ):
        if not processing_minutes:
            return [(start, start)], False
        if deadline and start >= deadline:
            return [(start, start)], True
        result = []
        remaining = processing_minutes
        current = start

        while remaining > 0:
            if deadline and deadline <= current:
                return result, True
            break_periods = []
            for b_start, b_end in self.list_breaktime:
                break_start = current.replace(hour=b_start.hour, minute=b_start.minute)
                break_end = current.replace(hour=b_end.hour, minute=b_end.minute)
                if break_end <= break_start:
                    continue
                break_periods.append((break_start, break_end))

            break_periods.sort()

            today_end = (current + timedelta(days=1)).replace(hour=0, minute=0)

            if self.is_day_off(current):
                interval_end = today_end
                minutes_available = int((interval_end - current).total_seconds() / 60)
                duration = min(remaining, minutes_available)
                if duration > 0:
                    current_end = current + timedelta(minutes=duration)
                    if deadline and current < deadline < current_end:
                        result.append((current, deadline))
                        return result, True
                    else:
                        result.append((current, current_end))
                    remaining -= duration
                    current += timedelta(minutes=duration)
                    continue

            pointer = current
            for break_start, break_end in break_periods:
                if pointer >= today_end:
                    break

                if pointer < break_start:
                    interval_end = min(break_start, today_end)
                    duration = int((interval_end - pointer).total_seconds() / 60)
                    if duration > 0:
                        used = min(duration, remaining)
                        pointer_end = pointer + timedelta(minutes=used)
                        if deadline and current < deadline < pointer_end:
                            result.append((current, deadline))
                            return result, True
                        else:
                            result.append((pointer, pointer_end))
                        remaining -= used
                        pointer += timedelta(minutes=used)
                        if remaining == 0:
                            return result, False

                if pointer < break_end:
                    pointer = break_end

            if pointer < today_end:
                duration = int((today_end - pointer).total_seconds() / 60)
                used = min(duration, remaining)
                pointer_end = pointer + timedelta(minutes=used)
                if deadline and current < deadline < pointer_end:
                    result.append((current, deadline))
                    return result, True
                else:
                    result.append((pointer, pointer_end))
                remaining -= used
                pointer += timedelta(minutes=used)

            current = today_end
        return result, False

    def are_shifts_connected(self, start_time, end_time, is_work_break_time=False):
        if start_time == end_time:
            return True
        if (
            (start_time.time(), end_time.time()) in self.list_breaktime
            and start_time.date() == end_time.date()
            and not is_work_break_time
        ):
            return True
        elif (
            start_time.time() == self.WORKING_HOURS_END
            and end_time.time() == self.WORKING_HOURS_START
        ):
            if self.is_all_day_off(start_time, end_time):
                return True
            else:
                return False
        return False


class TimeComputationExtention(TimeComputation):
    def resource_filter_linker(
        self,
        process_unit,
        dict_force_start,
        time_grouper,
        time_grouper_machine,
        deadline=None,
    ):
        schedule_predict_step_dict = defaultdict(lambda: defaultdict(list))
        schedule_predict_saver_dict = defaultdict(dict)
        list_valid_ids = []
        list_final_ids = []
        step = constants.PRE_PROCESS_STEP
        p_time = process_unit["timing_details"][step]
        list_final_ids_tmp = []
        for id, start_process in dict_force_start.items():
            worker_id, machine_id, _ = map(lambda x: int(float(x)), id.split("___"))
            is_auto = self.factory_rs_info["dict_machine_auto"].get(machine_id, False)
            allocated_time_list_tmp, exceed_end_max = self.generate_working_intervals(
                start_dt=start_process,
                min_total_minutes=p_time,
                is_work_break_time=process_unit["is_work_break_time"],
                deadline=deadline,
            )
            if exceed_end_max:
                list_final_ids_tmp.append(id)
            schedule_predict_step_dict[step][id].append(allocated_time_list_tmp)
            schedule_predict_saver_dict[id][step] = allocated_time_list_tmp
        _, per_group_verdict = self.validate_queries_inside_availability(
            time_grouper_qr=schedule_predict_step_dict[step],
            time_grouper=time_grouper,
            deadline=deadline,
            auto=False,
        )
        per_group_verdict = per_group_verdict.filter(
            pl.col("all_queries_contained")
        ).with_columns(
            (pl.col("id") + "___" + pl.col("group").cast(pl.Utf8)).alias("id_group")
        )
        list_valid_ids.extend(per_group_verdict["id_group"].to_list())
        list_final_ids.extend(list(set(list_valid_ids) & set(list_final_ids_tmp)))
        list_valid_ids = list(set(list_valid_ids) - set(list_final_ids))
        for step, p_time in process_unit["timing_details"].items():
            if step == constants.PRE_PROCESS_STEP:
                continue
            list_final_ids_tmp = []
            for id in list_valid_ids:
                _, machine_id, _ = map(lambda x: int(float(x)), id.split("___"))
                is_auto = self.factory_rs_info["dict_machine_auto"].get(
                    machine_id, False
                )
                current_end = schedule_predict_saver_dict[id][
                    constants.PREVIOUS_PROCESS_STEP[step]
                ][-1][-1]
                if (
                    process_unit["is_nightshift_expected"]
                    and step == constants.PROCESSING_STEP
                ):
                    if not process_unit["is_work_break_time"]:
                        allocated_time_list_tmp, exceed_end_max = (
                            self.allocate_processing_time_with_breaks(
                                start=current_end,
                                processing_minutes=p_time,
                                deadline=deadline,
                            )
                        )
                    else:
                        exceed_end_max = False
                        end_mid_process = current_end + timedelta(minutes=p_time)
                        if deadline and current_end <= deadline <= end_mid_process:
                            allocated_time_list_tmp = [(current_end, deadline)]
                            exceed_end_max = True
                        else:
                            allocated_time_list_tmp = [(current_end, end_mid_process)]
                    if process_unit["timing_details"][constants.POST_PROCESS_STEP] > 0:
                        end_mid_process = self.shift_end_to_next_holiday(
                            allocated_time_list_tmp[-1][-1],
                            process_unit["is_work_break_time"],
                        )
                        if (
                            deadline
                            and allocated_time_list_tmp[-1][0]
                            <= deadline
                            <= end_mid_process
                        ):
                            allocated_time_list_tmp[-1] = (
                                allocated_time_list_tmp[-1][0],
                                deadline,
                            )
                            exceed_end_max = True
                        else:
                            allocated_time_list_tmp[-1] = (
                                allocated_time_list_tmp[-1][0],
                                end_mid_process,
                            )
                else:
                    allocated_time_list_tmp, exceed_end_max = (
                        self.generate_working_intervals(
                            start_dt=current_end,
                            min_total_minutes=p_time,
                            is_work_break_time=process_unit["is_work_break_time"],
                            deadline=deadline,
                        )
                    )
                if exceed_end_max:
                    list_final_ids_tmp.append(id)
                step_auto = (
                    "processing_auto"
                    if is_auto and step == constants.PROCESSING_STEP
                    else step
                )
                schedule_predict_step_dict[step_auto][f"{id}"].append(
                    allocated_time_list_tmp
                )
                schedule_predict_saver_dict[f"{id}"][step] = allocated_time_list_tmp
            list_valid_ids = []
            if (
                step == constants.PROCESSING_STEP
                and schedule_predict_step_dict["processing_auto"]
            ):
                time_grouper_machine = self.avail_time_grouper(time_grouper_machine)
                time_grouper_local = time_grouper_machine
                if process_unit["is_nightshift_expected"]:
                    time_grouper_local = self.nightshift_timegrouper_convert(
                        time_grouper_machine
                    )
                _, per_group_verdict = self.validate_queries_inside_availability(
                    time_grouper_qr=schedule_predict_step_dict["processing_auto"],
                    time_grouper=time_grouper_local,
                    deadline=deadline,
                    auto=True,
                )
                per_group_verdict = per_group_verdict.filter(
                    pl.col("all_queries_contained")
                ).with_columns(
                    (pl.col("id") + "___" + pl.col("group").cast(pl.Utf8)).alias(
                        "id_group"
                    )
                )
                list_valid_ids.extend(per_group_verdict["id_group"].to_list())
            if schedule_predict_step_dict[step]:
                time_grouper_local = time_grouper
                if step == constants.PROCESSING_STEP:
                    if process_unit["is_nightshift_expected"]:
                        time_grouper_local = self.nightshift_timegrouper_convert(
                            time_grouper
                        )
                _, per_group_verdict = self.validate_queries_inside_availability(
                    time_grouper_qr=schedule_predict_step_dict[step],
                    time_grouper=time_grouper_local,
                    deadline=deadline,
                    auto=False,
                )
                per_group_verdict = per_group_verdict.filter(
                    pl.col("all_queries_contained")
                ).with_columns(
                    (pl.col("id") + "___" + pl.col("group").cast(pl.Utf8)).alias(
                        "id_group"
                    )
                )
                list_valid_ids.extend(per_group_verdict["id_group"].to_list())
            list_final_ids.extend(list(set(list_valid_ids) & set(list_final_ids_tmp)))
            list_valid_ids = list(set(list_valid_ids) - set(list_final_ids_tmp))
        return list_valid_ids + list_final_ids, schedule_predict_step_dict

    def build_enough_time_list_unmanned_process_nightshift_priority_linker(
        self,
        process_unit,
        time_grouper,
        time_grouper_machine,
        total_allocated_time,
        overtime,
    ):
        decision_dict = {}
        dict_operating_rate_worker = {}
        dict_operating_rate_machine = {}
        list_valid_ids, _ = self.resource_filter(
            process_unit, time_grouper, time_grouper_machine
        )
        for id, ts_list in time_grouper.items():
            worker_id = int(float(id.split("___")[0]))
            process_id = int(float(id.split("___")[1]))
            ts_list_machine = time_grouper_machine[process_id]
            allocated_time_dict = defaultdict(list)
            operating_rate_worker = dict_operating_rate_worker.setdefault(
                worker_id,
                self.get_operating_rate(
                    rs_id=worker_id,
                    total_allocated_time=total_allocated_time,
                    type="worker",
                ),
            )
            operating_rate_machine = dict_operating_rate_machine.setdefault(
                process_id,
                self.get_operating_rate(
                    rs_id=process_id,
                    total_allocated_time=total_allocated_time,
                    type="machine",
                ),
            )
            for index in range(len(ts_list)):
                if f"{id}___{index}" not in list_valid_ids:
                    continue
                previous_processing_time = process_unit["timing_details"][
                    constants.PRE_PROCESS_STEP
                ]
                mid_processing_time = process_unit["timing_details"][
                    constants.PROCESSING_STEP
                ]
                post_processing_time = process_unit["timing_details"][
                    constants.POST_PROCESS_STEP
                ]
                grouped_time = ts_list[index]
                if not len(grouped_time):
                    continue
                grouped_time_df = pd.DataFrame(
                    data=grouped_time, columns=["start", "end"]
                )
                grouped_time_df["diff"] = (
                    grouped_time_df["end"] - grouped_time_df["start"]
                )
                grouped_time_df["cumsum"] = grouped_time_df["diff"].cumsum()

                grouped_time_df = grouped_time_df[
                    (
                        grouped_time_df["cumsum"]
                        >= timedelta(minutes=previous_processing_time)
                    )
                    & (grouped_time_df["end"].dt.hour >= self.WORKING_HOURS_END.hour)
                ]
                if grouped_time_df.empty:
                    continue
                available_pre_process = grouped_time[: (grouped_time_df.index[0] + 1)]
                if previous_processing_time > 0:
                    while previous_processing_time > 0 and available_pre_process:
                        start_time, end_time = available_pre_process[-1]
                        time_difference = (end_time - start_time).total_seconds() / 60
                        if time_difference >= previous_processing_time:
                            start_pre_process = end_time - timedelta(
                                minutes=previous_processing_time
                            )
                            previous_processing_time = 0
                            allocated_time_dict[constants.PRE_PROCESS_STEP].append(
                                (start_pre_process, end_time)
                            )
                            if start_pre_process == start_time:
                                available_pre_process.pop(-1)
                        else:
                            allocated_time_dict[constants.PRE_PROCESS_STEP].append(
                                available_pre_process.pop(-1)
                            )
                            previous_processing_time -= time_difference
                else:
                    allocated_time_dict[constants.PRE_PROCESS_STEP] = [
                        (available_pre_process[-1][-1], available_pre_process[-1][-1])
                    ]
                allocated_time_dict[constants.PRE_PROCESS_STEP] = list(
                    reversed(allocated_time_dict[constants.PRE_PROCESS_STEP])
                )

                is_mid_process_not_suitable = False
                if mid_processing_time > 0:
                    end_mid_process = allocated_time_dict[constants.PRE_PROCESS_STEP][
                        -1
                    ][-1] + timedelta(minutes=mid_processing_time)
                    allocated_time_dict[constants.PROCESSING_STEP] = [
                        (
                            allocated_time_dict[constants.PRE_PROCESS_STEP][-1][-1],
                            end_mid_process,
                        )
                    ]
                else:
                    allocated_time_dict[constants.PROCESSING_STEP] = [
                        (
                            allocated_time_dict[constants.PRE_PROCESS_STEP][-1][-1],
                            allocated_time_dict[constants.PRE_PROCESS_STEP][-1][-1],
                        )
                    ]
                start_mid_process = allocated_time_dict[constants.PROCESSING_STEP][0][0]
                end_mid_process = allocated_time_dict[constants.PROCESSING_STEP][-1][-1]
                for gr_time_machine in ts_list_machine:
                    s_machine = gr_time_machine[0]
                    e_machine = gr_time_machine[1]
                    is_mid_process_not_suitable = True
                    if s_machine <= start_mid_process <= end_mid_process <= e_machine:
                        is_mid_process_not_suitable = False
                        break
                if is_mid_process_not_suitable:
                    continue
                end_mid_process = self.shift_end_to_next_holiday(
                    end_mid_process, process_unit["is_work_break_time"]
                )
                allocated_time_dict[constants.PROCESSING_STEP][-1] = (
                    allocated_time_dict[constants.PROCESSING_STEP][-1][0],
                    end_mid_process,
                )
                time_ranges = ts_list[index]
                is_post_process_not_suitable = False
                if not post_processing_time == 0:
                    while end_mid_process >= time_ranges[-1][-1]:
                        index += 1
                        if index > len(ts_list) - 1:
                            is_post_process_not_suitable = True
                            break
                        time_ranges = ts_list[index]
                    while time_ranges:
                        start_time, end_time = time_ranges[0]
                        if end_time <= end_mid_process:
                            time_ranges.pop(0)
                        else:
                            time_ranges[0] = (end_mid_process, end_time)
                            break
                    else:
                        is_post_process_not_suitable = True
                    if is_post_process_not_suitable:
                        continue
                    is_post_process_not_suitable = False
                    if post_processing_time > 0:
                        while post_processing_time > 0 and time_ranges:
                            start_time, end_time = time_ranges[0]
                            time_difference = end_time - start_time

                            if (
                                time_difference.total_seconds()
                                >= post_processing_time * 60
                            ):
                                end_post_process = start_time + timedelta(
                                    minutes=post_processing_time
                                )
                                post_processing_time = 0
                                allocated_time_dict[constants.POST_PROCESS_STEP].append(
                                    (start_time, end_post_process)
                                )
                                if end_post_process == end_time:
                                    time_ranges.pop(0)
                                else:
                                    time_ranges[0] = (end_post_process, end_time)
                            else:
                                if len(time_ranges):
                                    allocated_time_dict[
                                        constants.POST_PROCESS_STEP
                                    ].append(time_ranges.pop(0))
                                    post_processing_time -= (
                                        time_difference.total_seconds() // 60
                                    )
                                    if not len(time_ranges) and post_processing_time:
                                        is_post_process_not_suitable = True
                                        break
                                else:
                                    is_post_process_not_suitable = True
                                    break
                    else:
                        allocated_time_dict[constants.POST_PROCESS_STEP].append(
                            (time_ranges[0][0], time_ranges[0][0])
                        )

                    if is_post_process_not_suitable:
                        continue
                else:
                    if end_mid_process > time_ranges[-1][-1]:
                        continue
                    allocated_time_dict[constants.POST_PROCESS_STEP].append(
                        (end_mid_process, end_mid_process)
                    )
                end_process = allocated_time_dict[constants.POST_PROCESS_STEP][-1][-1]
                # build decision dict
                decision_dict[f"{id}___{index}"] = {
                    "end": end_process,
                    "overtime_hours": (
                        self.get_overtime_hours(
                            allocated_time_dict[constants.PRE_PROCESS_STEP][0][0],
                            end_process,
                        )
                        if overtime
                        else 0
                    ),
                    "operating_rate": operating_rate_worker + operating_rate_machine,
                    "allocated_time_dict": allocated_time_dict,
                    "is_not_auto": not self.factory_rs_info["dict_machine_auto"].get(
                        process_id, False
                    ),
                    "night_time": self.compute_night_time_hours(
                        allocated_time_dict[constants.PROCESSING_STEP]
                    ),
                }
        return decision_dict

    def build_allocated_time_list_for_linker(
        self,
        process_unit,
        list_valid_ids,
        schedule_predict_step_dict,
        total_allocated_time,
        overtime,
        deadline=None,
    ):
        decision_dict = {}
        best_end_process = None
        dict_operating_rate_worker = {}
        dict_operating_rate_machine = {}
        dict_start_of_ids = {}
        for id in list_valid_ids:
            dict_start_of_ids[id] = schedule_predict_step_dict[
                constants.PRE_PROCESS_STEP
            ][id][0][0][0]
        for id, start_process in sorted(dict_start_of_ids.items(), key=lambda x: x[1]):
            allocated_time_dict = defaultdict()
            current_end = start_process
            for step, p_time in process_unit["timing_details"].items():
                if (
                    process_unit["is_nightshift_expected"]
                    and step == constants.PROCESSING_STEP
                ):
                    if not process_unit["is_work_break_time"]:
                        allocated_time_list_tmp, _ = (
                            self.allocate_processing_time_with_breaks(
                                start=current_end,
                                processing_minutes=p_time,
                                deadline=deadline,
                            )
                        )
                    else:
                        end_mid_process = current_end + timedelta(minutes=p_time)
                        if deadline and current_end <= deadline <= end_mid_process:
                            allocated_time_list_tmp = [
                                (current_end, deadline),
                                (deadline, end_mid_process),
                            ]
                        else:
                            allocated_time_list_tmp = [(current_end, end_mid_process)]
                    if process_unit["timing_details"][constants.POST_PROCESS_STEP] > 0:
                        end_mid_process = self.shift_end_to_next_holiday(
                            allocated_time_list_tmp[-1][-1],
                            process_unit["is_work_break_time"],
                        )
                        if (
                            deadline
                            and allocated_time_list_tmp[-1][0]
                            <= deadline
                            <= end_mid_process
                        ):
                            allocated_time_list_tmp[-1] = (
                                allocated_time_list_tmp[-1][0],
                                deadline,
                            )
                            allocated_time_list_tmp.append((deadline, end_mid_process))
                        else:
                            allocated_time_list_tmp[-1] = (
                                allocated_time_list_tmp[-1][0],
                                end_mid_process,
                            )
                else:
                    allocated_time_list_tmp, _ = self.generate_working_intervals(
                        start_dt=current_end,
                        min_total_minutes=p_time,
                        is_work_break_time=process_unit["is_work_break_time"],
                    )
                allocated_time_dict[step] = allocated_time_list_tmp
                current_end = allocated_time_list_tmp[-1][-1]

            end_process = self.max_end_date_in_dict(allocated_time_dict)
            if best_end_process is not None and end_process > best_end_process:
                continue
            worker_id, machine_id, _ = map(lambda x: int(float(x)), id.split("___"))
            is_auto = self.factory_rs_info["dict_machine_auto"].get(machine_id, False)
            operating_rate_worker = dict_operating_rate_worker.setdefault(
                worker_id,
                self.get_operating_rate(
                    rs_id=worker_id,
                    total_allocated_time=total_allocated_time,
                    type="worker",
                ),
            )
            operating_rate_machine = dict_operating_rate_machine.setdefault(
                machine_id,
                self.get_operating_rate(
                    rs_id=machine_id,
                    total_allocated_time=total_allocated_time,
                    type="machine",
                ),
            )
            decision_dict[f"{id}"] = {
                "end": self.max_end_date_in_dict(allocated_time_dict),
                "overtime_hours": (
                    self.get_overtime_hours(
                        start_process, end_process, process_unit["is_work_break_time"]
                    )
                    if overtime
                    else 0
                ),
                "operating_rate": operating_rate_worker + operating_rate_machine,
                "is_not_auto": not is_auto,
                "allocated_time_dict": allocated_time_dict,
            }
            best_end_process = end_process
        return decision_dict

    def schedule_core_linker(
        self,
        process_unit,
        decision_dict_child,
        factory_resources,
        deadline,
        overtime=False,
        night_time_prioritize=False,
        keep_deadline=True,
        time_grouper_machine=None,
        force_start=None,
    ):
        for k, t_range in factory_resources.items():
            temp_time_dict = []
            for s, e in t_range:
                temp_time_dict.append(
                    (
                        max(
                            s,
                            s.replace(
                                hour=self.WORKING_HOURS_START.hour,
                                minute=self.WORKING_HOURS_START.minute,
                            ),
                        ),
                        min(
                            e,
                            e.replace(
                                hour=self.WORKING_HOURS_END.hour,
                                minute=self.WORKING_HOURS_END.minute,
                            ),
                        ),
                    )
                )
            factory_resources[k] = temp_time_dict
        time_grouper = self.avail_time_grouper(
            factory_resources, process_unit["is_work_break_time"]
        )
        total_allocated_time_in_period = self.get_diff_hours(
            self.dt_current_date,
            deadline,
            is_work_break_time=process_unit["is_work_break_time"],
        )
        if keep_deadline and not time_grouper:
            return {}
        if decision_dict_child:
            dict_force_start = {}
            if force_start:
                for id, ts_list in time_grouper.items():
                    dict_force_start[f"{id}___0"] = force_start
            else:
                for id, value in decision_dict_child.items():
                    dict_force_start[id] = self.max_end_date_in_dict(
                        value["allocated_time_dict"]
                    )
                    for id_new, ts_list in time_grouper.items():
                        dict_force_start[f"{id_new}___0"] = self.max_end_date_in_dict(
                            value["allocated_time_dict"]
                        )
            list_valid_ids, schedule_predict_step_dict = self.resource_filter_linker(
                process_unit=process_unit,
                dict_force_start=dict_force_start,
                time_grouper=time_grouper,
                time_grouper_machine=time_grouper_machine,
                deadline=deadline,
            )
        else:
            list_valid_ids, schedule_predict_step_dict = self.resource_filter(
                process_unit=process_unit,
                time_grouper=time_grouper,
                time_grouper_machine=time_grouper_machine,
                deadline=deadline,
            )
        if night_time_prioritize or False:
            decision_dict_tmp = (
                self.build_enough_time_list_unmanned_process_nightshift_priority_linker(
                    process_unit=process_unit,
                    total_allocated_time=total_allocated_time_in_period,
                    overtime=overtime,
                )
            )
        else:
            decision_dict_tmp = self.build_allocated_time_list_for_linker(
                process_unit=process_unit,
                list_valid_ids=list_valid_ids,
                schedule_predict_step_dict=schedule_predict_step_dict,
                total_allocated_time=total_allocated_time_in_period,
                overtime=overtime,
                deadline=None if keep_deadline else deadline,
            )
        return decision_dict_tmp

    def extract_start_end(self, process_info):
        alloc = process_info["allocated_time_dict"]
        start = alloc["pre_processing"][0][0]
        end = alloc["post_processing"][-1][1]
        return start, end

    def build_chains(self, process_dict):
        intervals = {
            pid: self.extract_start_end(info) for pid, info in process_dict.items()
        }
        chains = []
        used = set()
        for pid, (s, e) in intervals.items():
            if pid in used:
                continue
            chain = [pid]
            used.add(pid)
            while True:
                found = False
                for nxt, (s2, e2) in intervals.items():
                    if nxt not in used and self.are_shifts_connected(e, s2):
                        chain.append(nxt)
                        used.add(nxt)
                        e = e2
                        found = True
                        break
                if not found:
                    break
            chains.append(chain)
        return chains


if __name__ == "__main__":

    time_computer = TimeComputation(
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
        holiday={"days": ["2025-10-03"]},  # Friday, Oct 3rd is a holiday
        list_breaktime=[((10, 0), (10, 15))],  # 15-min tea break
        factory_rs_info={
            "dict_machine_auto": {101: False, 102: True}
        },  # Machine 101: Manual, 102: Auto
        customer_rs=None,
        timezone=None,
    )

    # Set a shared attribute needed for the method
    time_computer.dict_total_worked_time = {"worker": {1: 100}, "machine": {101: 50}}
