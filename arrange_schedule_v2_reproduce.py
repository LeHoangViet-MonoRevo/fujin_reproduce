from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import polars as pl

import logger
from constants import constants
from processmapbuilder_reproduce import ProcessMapBuilder
from timecomputation_reproduce import TimeComputationExtention

logger = logger.get_logger(logger_name="SERVICE_ARRANGE_SCHEDULE")


class ArrangeScheduleService(TimeComputationExtention):
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
        process_list,
        start_planning_time,
        end_planning_time,
        settings,
        factory_rs_info,
        customer_rs,
        timezone,
        lead_time=None,
    ):
        super(ArrangeScheduleService, self).__init__(
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
        )
        self.process_map_object = ProcessMapBuilder(process_list, settings.same_shape)
        self.process_map_object.build_dependency()
        self.dt_current_date = datetime.strptime(
            start_planning_time, constants.DT_FORMAT
        ).replace(second=0)
        self.dt_delivery_date = datetime.strptime(
            end_planning_time, constants.DT_FORMAT
        ).replace(second=0)
        self.process_list = process_list
        self.settings = settings
        self.holiday = holiday
        self.lead_time = lead_time
        self.factory_rs_info = factory_rs_info
        self.customer_rs = customer_rs
        self.timezone = timezone
        self.batch_return = False

    def get_factory_resources_list(self, process_unit, worker_details_df):
        if not process_unit["is_internal"]:
            return [process_unit["worker_id"]], [process_unit["machine_id"]], {}
        dict_worker_matching = {}
        list_worker = []
        list_machine = []
        list_total_worker = worker_details_df["worker_id"].to_list()
        if process_unit["process_category"]:
            if process_unit["machine_id"]:
                list_machine = [process_unit["machine_id"]]
                if process_unit["worker_id"]:
                    list_worker = [process_unit["worker_id"]]
                else:
                    list_worker = self.factory_rs_info[
                        "dict_worker_group_to_worker"
                    ].get(process_unit["worker_group"], list_total_worker)
                    list_worker_by_machine = self.factory_rs_info[
                        "dict_machine_to_worker"
                    ].get(process_unit["machine_id"], list_total_worker)
                    list_worker = list(set(list_worker) & set(list_worker_by_machine))
            else:
                if process_unit["worker_id"]:
                    raise ValueError("Invalid information")
                else:
                    list_machine = list(
                        set(
                            self.factory_rs_info[
                                "dict_process_category_to_machine"
                            ].get(
                                process_unit["process_category"],
                                self.factory_rs_info["list_machine_active"],
                            )
                        )
                    )
                    list_worker_by_group = self.factory_rs_info[
                        "dict_worker_group_to_worker"
                    ].get(process_unit["worker_group"], list_total_worker)
                    for machine in list_machine:
                        list_worker_by_machine = self.factory_rs_info[
                            "dict_machine_to_worker"
                        ].get(machine, list_total_worker)
                        list_worker_by_machine = list(
                            set(list_worker_by_group) & set(list_worker_by_machine)
                        )
                        list_worker.extend(list_worker_by_machine)
                        dict_worker_matching.setdefault(machine, []).extend(
                            list_worker_by_machine
                        )
                    list_worker = list(set(list_worker))
                    dict_worker_matching = {
                        key: list(set(values))
                        for key, values in dict_worker_matching.items()
                    }
        else:
            raise ValueError("Process category can't be left blank")
        if not len(list_worker):
            raise RuntimeError("We dont have any suitable workers")
        if not len(list_machine):
            raise RuntimeError("We dont have any suitable process")
        return list_worker, list_machine, dict_worker_matching

    def get_worked_time_range_factory_resources(
        self,
        list_worker,
        list_machine,
        worker_details_df,
        working_time_available_worker_df,
        machine_details_df,
        working_time_available_machine_df,
        current_date,
        delivery_date,
    ):
        for type in ("worker", "machine"):
            if type == "worker":
                factory_resources_details_df = worker_details_df
                working_time_available_factory_resources_df = (
                    working_time_available_worker_df
                )
                list_factory_resources = list_worker
            else:
                factory_resources_details_df = machine_details_df
                working_time_available_factory_resources_df = (
                    working_time_available_machine_df
                )
                list_factory_resources = list_machine
            if len(list_factory_resources) == 0:
                raise RuntimeError("We dont have any suitable machine")
            factory_resources_details_df = pd.concat(
                [
                    factory_resources_details_df,
                    pd.DataFrame(
                        {
                            f"{type}_id": list(
                                set(list_factory_resources)
                                - set(factory_resources_details_df[f"{type}_id"])
                            )
                        }
                    ),
                ],
                ignore_index=True,
            )
            factory_resources_details_df[f"{type}_id"] = factory_resources_details_df[
                f"{type}_id"
            ].astype(int)
            factory_resources_fit = factory_resources_details_df[
                (
                    factory_resources_details_df.eval(f"{type}_id").isin(
                        list_factory_resources
                    )
                )
            ]
            factory_resources_working_df = working_time_available_factory_resources_df[
                (
                    working_time_available_factory_resources_df.eval(f"{type}_id").isin(
                        list_factory_resources
                    )
                )
                & (
                    (
                        (
                            (
                                working_time_available_factory_resources_df[
                                    "start_date"
                                ]
                                < delivery_date
                            )
                            & (
                                working_time_available_factory_resources_df[
                                    "start_date"
                                ]
                                > current_date
                            )
                        )
                        | (
                            (
                                working_time_available_factory_resources_df["end_date"]
                                < delivery_date
                            )
                            & (
                                working_time_available_factory_resources_df["end_date"]
                                > current_date
                            )
                        )
                        | (
                            (
                                working_time_available_factory_resources_df[
                                    "start_date"
                                ]
                                <= current_date
                            )
                            & (
                                working_time_available_factory_resources_df["end_date"]
                                >= delivery_date
                            )
                        )
                    )
                )
            ]
            id_counts = (
                factory_resources_working_df[f"{type}_id"].value_counts().reset_index()
            )
            id_counts.columns = [f"{type}_id", "number_existing_process"]
            count_factory_resources = (
                factory_resources_fit[[f"{type}_id"]]
                .merge(id_counts, on=f"{type}_id", how="left")
                .fillna(0)
            )
            count_factory_resources = count_factory_resources.astype(
                {"number_existing_process": int}
            )

            factory_resources_working_df = factory_resources_working_df.sort_values(
                by=["start_date"]
            )
            if type == "worker":
                worker_working_df = factory_resources_working_df
                count_worker = count_factory_resources
            else:
                machine_working_df = factory_resources_working_df
                count_machine = count_factory_resources

        return worker_working_df, count_worker, machine_working_df, count_machine

    def get_resource_available_time(
        self, working_df, count_df, id_field, dt_current_date, dt_delivery_date
    ):
        dict_avail_time = {}
        type = id_field.split("_")[0]
        if type not in ("worker", "machine"):
            raise ValueError(
                'Work type is not valid, please enter "worker" or "machine"'
            )
        for _, row in count_df.iterrows():
            dict_avail_time[row[id_field]] = []
            time_temp = []
            if row.number_existing_process == 0:
                if type == "worker":
                    all_days = []
                    dt_current_date_temp = dt_current_date
                    dt_current_date_temp = dt_current_date_temp.replace(minute=0)
                    while dt_current_date_temp.date() <= dt_delivery_date.date():
                        all_days.append(dt_current_date_temp)
                        dt_current_date_temp += timedelta(days=1)
                    middle_dates = all_days[1:-1]
                    middle_dates = [x for x in middle_dates if not self.is_day_off(x)]
                    if (
                        dt_current_date.time() < self.WORKING_HOURS_END
                        and not self.is_day_off(dt_current_date)
                    ):
                        time_temp.append(
                            (
                                max(
                                    dt_current_date,
                                    dt_current_date.replace(
                                        hour=self.WORKING_HOURS_START.hour,
                                        minute=self.WORKING_HOURS_START.minute,
                                    ),
                                ),
                                dt_current_date.replace(
                                    hour=self.WORKING_HOURS_END.hour,
                                    minute=self.WORKING_HOURS_END.minute,
                                ),
                            )
                        )
                    for d in middle_dates:
                        time_temp.append(
                            (
                                d.replace(
                                    hour=self.WORKING_HOURS_START.hour,
                                    minute=self.WORKING_HOURS_START.minute,
                                ),
                                d.replace(
                                    hour=self.WORKING_HOURS_END.hour,
                                    minute=self.WORKING_HOURS_END.minute,
                                ),
                            )
                        )
                    if (
                        self.WORKING_HOURS_START
                        < dt_delivery_date.time()
                        <= self.WORKING_HOURS_END
                        and not self.is_day_off(dt_delivery_date)
                    ):
                        time_temp.append(
                            (
                                dt_delivery_date.replace(
                                    hour=self.WORKING_HOURS_START.hour,
                                    minute=self.WORKING_HOURS_START.minute,
                                ),
                                dt_delivery_date,
                            )
                        )
                else:
                    time_temp.append((dt_current_date, dt_delivery_date))
            else:
                allocated_slot = working_df[working_df[id_field] == row[id_field]]
                allocated_slot = list(
                    zip(allocated_slot.start_date, allocated_slot.end_date)
                )
                time_temp = self.get_available_time(
                    dt_current_date, dt_delivery_date, allocated_slot, type=type
                )
            dict_avail_time[row[id_field]] = time_temp
        dict_avail_time = {
            key: value for key, value in dict_avail_time.items() if value != []
        }
        return dict_avail_time

    def dict_to_pl(self, availability_dict, key_name: str) -> pl.DataFrame:
        records = [
            {key_name: str(k), f"start_of_{key_name}": s, f"end_of_{key_name}": e}
            for k, periods in availability_dict.items()
            for s, e in periods
        ]
        return pl.DataFrame(records)

    def explode_touched_days_pl(
        self,
        df: pl.DataFrame,
        start_col: str,
        end_col: str,
        out_col: str = "covered_day",
    ) -> pl.DataFrame:
        return (
            df.with_columns(
                [
                    pl.col(start_col).cast(pl.Datetime("ns")).alias("_s"),
                    pl.col(end_col).cast(pl.Datetime("ns")).alias("_e"),
                ]
            )
            .with_columns(
                [
                    (pl.col("_e") - pl.duration(nanoseconds=1)).alias("_e_adj"),
                ]
            )
            .with_columns(
                [
                    pl.date_ranges(
                        start=pl.col("_s").dt.truncate("1d"),
                        end=pl.col("_e_adj").dt.truncate("1d"),
                        interval="1d",
                        closed="both",
                    ).alias(out_col)
                ]
            )
            .explode(out_col)
            .drop(["_s", "_e", "_e_adj"])
        )

    def get_avail_total(self, worker_avail_time: dict, machine_avail_time: dict):
        if not worker_avail_time or not machine_avail_time:
            return {}

        list_tmp_workers = []
        list_tmp_machine = []
        for k, v in worker_avail_time.items():
            list_tmp_workers.append(len(v))
        for k, v in machine_avail_time.items():
            list_tmp_machine.append(len(v))
        if sum(list_tmp_workers) == 0 or sum(list_tmp_machine) == 0:
            return {}

        worker_df = self.dict_to_pl(worker_avail_time, "worker")
        machine_df = self.dict_to_pl(machine_avail_time, "machine_id")

        worker_df = worker_df.with_columns(
            [
                pl.col("start_of_worker").cast(pl.Datetime("ns")),
                pl.col("end_of_worker").cast(pl.Datetime("ns")),
            ]
        )
        machine_df = machine_df.with_columns(
            [
                pl.col("start_of_machine_id").cast(pl.Datetime("ns")),
                pl.col("end_of_machine_id").cast(pl.Datetime("ns")),
            ]
        )

        worker_days = self.explode_touched_days_pl(
            worker_df, "start_of_worker", "end_of_worker", "covered_day"
        )
        machine_days = self.explode_touched_days_pl(
            machine_df, "start_of_machine_id", "end_of_machine_id", "covered_day"
        )

        pairs = worker_days.join(machine_days, on="covered_day", how="inner")

        pairs = (
            pairs.with_columns(
                [
                    pl.max_horizontal("start_of_worker", "start_of_machine_id").alias(
                        "overlap_start"
                    ),
                    pl.min_horizontal("end_of_worker", "end_of_machine_id").alias(
                        "overlap_end"
                    ),
                ]
            )
            .filter(pl.col("overlap_start") < pl.col("overlap_end"))
            .select(["worker", "machine_id", "overlap_start", "overlap_end"])
        )

        grouped = pairs.group_by(["worker", "machine_id"]).agg(
            [
                pl.col("overlap_start").implode().alias("overlap_start"),
                pl.col("overlap_end").implode().alias("overlap_end"),
            ]
        )
        result = {
            f"{r['worker']}___{r['machine_id']}": list(
                zip(r["overlap_start"], r["overlap_end"])
            )
            for r in grouped.to_dicts()
        }
        return result

    def schedule_core(
        self,
        process_unit,
        factory_resources,
        deadline,
        overtime=False,
        night_time_prioritize=False,
        keep_deadline=True,
        time_grouper_machine=None,
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
            return [("", "", {}, {})]
        common_resource_args = dict(
            process_unit=process_unit,
            time_grouper=time_grouper,
            time_grouper_machine=time_grouper_machine,
            total_allocated_time=total_allocated_time_in_period,
        )
        if night_time_prioritize:
            worker_id, machine_id, allocated_time_list = (
                self.build_enough_time_list_unmanned_process_nightshift_priority(
                    **common_resource_args, overtime=overtime
                )
            )
        else:
            worker_id, machine_id, allocated_time_list = self.build_allocated_time_list(
                **common_resource_args,
                overtime=overtime,
                deadline=None if keep_deadline else deadline,
            )
        return [
            (worker_id, machine_id, process_unit["process_id"], allocated_time_list)
        ]

    # def sort_priority(self, pid, process_dict_query):
    #     worker_id = process_dict_query[pid]['worker_id']
    #     automatic_work = process_dict_query[pid]['automatic_work']
    #     if worker_id != 0 and automatic_work:
    #         return 0
    #     elif worker_id != 0 and not automatic_work:
    #         return 1
    #     elif worker_id == 0 and automatic_work:
    #         return 2
    #     else:
    #         return 3

    def get_reasonable_resources(
        self,
        process_unit,
        worker_details_virtual_df,
        working_time_available_worker_df,
        machine_details_virtual_df,
        working_time_available_machine_df,
        current_date,
        delivery_date,
    ):
        list_worker, list_machine, dict_worker_matching = (
            self.get_factory_resources_list(
                process_unit=process_unit, worker_details_df=worker_details_virtual_df
            )
        )
        worker_working_df, count_worker, machine_working_df, count_machine = (
            self.get_worked_time_range_factory_resources(
                list_worker=list_worker,
                list_machine=list_machine,
                worker_details_df=worker_details_virtual_df,
                working_time_available_worker_df=working_time_available_worker_df,
                machine_details_df=machine_details_virtual_df,
                working_time_available_machine_df=working_time_available_machine_df,
                current_date=current_date,
                delivery_date=delivery_date,
            )
        )
        worker_avail_time, machine_avail_time = [
            self.get_resource_available_time(
                working_df=working_df,
                count_df=count_df,
                dt_current_date=current_date,
                dt_delivery_date=delivery_date,
                id_field=id_field,
            )
            for working_df, count_df, id_field in zip(
                (worker_working_df, machine_working_df),
                (count_worker, count_machine),
                ("worker_id", "machine_id"),
            )
        ]
        if not process_unit["is_work_break_time"]:
            for x in self.list_breaktime:
                worker_avail_time = self.rework_worker_avail_time_with_breaktime(
                    worker_avail_time, x[0], x[1]
                )
        worker_avail_time = {
            key: value for key, value in worker_avail_time.items() if value != []
        }
        avail_time = self.get_avail_total(worker_avail_time, machine_avail_time)
        return avail_time, machine_avail_time, dict_worker_matching

    def card_processing(
        self,
        working_time_available_worker_virtual_df,
        working_time_available_machine_virtual_df,
    ):
        process_list_tmp = []
        for process_unit in self.process_list:
            with self.adjust_schedule_time(overtime=process_unit["overtime"]):
                if (
                    process_unit["status"] not in constants.FIXED_STATUS
                    or not process_unit["machining_time"]
                ):
                    process_list_tmp.append(process_unit)
                    continue
                logger.info(f"{process_unit = }")
                allocated_time_dict = defaultdict(list)
                start_process = datetime.strptime(
                    process_unit["actual_start_at"], constants.DT_FORMAT
                ).replace(second=0)
                for step, p_time in process_unit["timing_details"].items():
                    if (
                        process_unit["is_nightshift_expected"]
                        and step == constants.PROCESSING_STEP
                    ):
                        if not process_unit["is_work_break_time"]:
                            allocated_time_list_tmp, _ = (
                                self.allocate_processing_time_with_breaks(
                                    start_process, p_time
                                )
                            )
                        else:
                            end_mid_process = start_process + timedelta(minutes=p_time)
                            allocated_time_list_tmp = [(start_process, end_mid_process)]
                        if (
                            process_unit["timing_details"][constants.POST_PROCESS_STEP]
                            > 0
                        ):
                            end_mid_process = self.shift_end_to_next_holiday(
                                allocated_time_list_tmp[-1][-1],
                                process_unit["is_work_break_time"],
                            )
                            allocated_time_list_tmp[-1] = (
                                allocated_time_list_tmp[-1][0],
                                end_mid_process,
                            )
                    else:
                        allocated_time_list_tmp, _ = self.generate_working_intervals(
                            start_dt=start_process,
                            min_total_minutes=p_time,
                            is_work_break_time=process_unit["is_work_break_time"],
                        )
                    allocated_time_dict[step] = allocated_time_list_tmp
                    start_process = self.max_end_date_in_dict(allocated_time_dict)
            logger.info(f"{process_unit['worker_id'] = }")
            logger.info(f"{process_unit['machine_id'] = }")
            worker_allocated_time_list = [v for _, v in allocated_time_dict.items()]
            worker_allocated_time_list = [
                item for sublist in worker_allocated_time_list for item in sublist
            ]
            worker_allocated_time_list = [
                x for x in worker_allocated_time_list if x[0] != x[1]
            ]
            logger.info(f"{worker_allocated_time_list = }")
            process_unit["start_at"] = worker_allocated_time_list[0][0].strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            process_unit["end_at"] = worker_allocated_time_list[-1][1].strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            process_list_tmp.append(process_unit)
        self.process_list = process_list_tmp
        return self.process_list

    def __push_results_to_db(self, doc):
        # total_card = doc.get("total_card")

        # self.buffer_temp_task_list.append(doc)
        # self.push_counter += 1
        # if self.push_counter == total_card:
        #     self.save_result_to_db(docs=self.buffer_temp_task_list)
        # elif len(self.buffer_temp_task_list) >= 50:
        #     self.save_result_to_db(docs=self.buffer_temp_task_list)
        #     self.buffer_temp_task_list = []
        pass

    def save_result_to_db(self, docs):
        # transformed_data = []
        # for doc in docs:
        #     process_schedule = doc.get("process_schedule")
        #     process_id = doc.get("process_id")
        #     logs = doc.get("logs")
        #     status = doc.get("status")
        #     total_card = doc.get("total_card")
        #     request_id = doc.get("request_id")
        #     production_lot_id, process_id_splited = process_id.split('___')
        #     p = process_schedule['process_time'][0]
        #     data = {
        #         "results": {
        #             "planning_process_id": int(process_id_splited),
        #             "production_lot_id": int(production_lot_id),
        #             "start_date_time": datetime.strptime(str(p[0]), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S+09:00'),
        #             "worker_id": int(process_schedule['worker_id']),
        #             "process_id": int(process_schedule['machine_id']),
        #             "is_overtime_expected": process_schedule.get('is_overtime_expected', False),
        #             "is_nightshift_expected": process_schedule.get('is_nightshift_expected', False)
        #         },
        #         "logs": logs,
        #         "status": status,
        #         "total_card": total_card,
        #         "created_at": timestamp(),
        #         "object_key": f"{request_id}##{int(process_id_splited)}",
        #         "request_id": request_id
        #     }
        #     transformed_data.append(data)
        # elasticsearch_db.bulk_insert_vector(
        #         indice_name=constants.ESLATICSEARCH_RESULTS_TASK_LIST_INDICE,
        #         docs=transformed_data,
        #         id_column="object_key",
        #         routing_key="request_id"
        #     )
        pass

    def build_schedule(
        self,
        worker_details_df,
        working_time_available_worker_df,
        machine_details_df,
        working_time_available_machine_df,
        process_df,
    ):
        # This one is the core function of this class
        logger.warning("=" * 50)
        working_time_available_worker_df["start_date"] = pd.to_datetime(
            pd.to_datetime(working_time_available_worker_df["start_date"]).dt.strftime(
                constants.DT_FORMAT
            )
        )
        working_time_available_worker_df["end_date"] = pd.to_datetime(
            pd.to_datetime(working_time_available_worker_df["end_date"]).dt.strftime(
                constants.DT_FORMAT
            )
        )

        working_time_available_machine_df["start_date"] = pd.to_datetime(
            pd.to_datetime(working_time_available_machine_df["start_date"]).dt.strftime(
                constants.DT_FORMAT
            )
        )
        working_time_available_machine_df["end_date"] = pd.to_datetime(
            pd.to_datetime(working_time_available_machine_df["end_date"]).dt.strftime(
                constants.DT_FORMAT
            )
        )

        self.process_list = self.card_processing(
            working_time_available_worker_virtual_df=working_time_available_worker_df,
            working_time_available_machine_virtual_df=working_time_available_machine_df,
        )

        worker_details_virtual_df = worker_details_df.copy()
        machine_details_virtual_df = machine_details_df.copy()

        working_time_available_worker_virtual_df = (
            working_time_available_worker_df.copy()
        )
        working_time_available_machine_virtual_df = (
            working_time_available_machine_df.copy()
        )

        process_schedule = {}
        process_schedule_human = {}

        self.total_hours = self.get_diff_hours(
            self.dt_current_date, self.dt_delivery_date
        )

        process_graph = self.process_map_object.build_process_graph_root()
        process_dict_query = {}
        a = []
        total_card = 0

        for process in self.process_list:
            a.append(process["machining_time"])
            process_dict_query[process["process_id"]] = process
            if self.batch_return:
                if not process["fixed"]:
                    total_card += 1

        # process_graph_auto = []
        # for layer in process_graph:
        #     process_graph_tmp = {}
        #     process_with_auto = sorted(layer['process_id'], key=lambda pid: self.sort_priority(pid, process_dict_query))
        #     process_graph_tmp['layout'] = layer['layout']
        #     process_graph_tmp['process_id'] = process_with_auto
        #     process_graph_auto.append(process_graph_tmp)
        # final_deadline_list = sorted(list(process_df['final_deadline'].unique()))
        for l_index, layer in enumerate(process_graph):
            have_same_shape = (
                True if layer["layout"] == 1 and self.settings.same_shape else False
            )
            for cluster in layer["process_id"]:
                for process in cluster:
                    process_unit_main = process_dict_query[process]
                    have_same_shape = (
                        False
                        if process_unit_main["product_shape"]["shape"]
                        == constants.OTHER_PRODUCT_SHAPE_NAME["shape"]
                        else have_same_shape
                    )
                    if (
                        process_unit_main["process_id"]
                        in self.process_map_object.COMPLETE_PROCESS_ID
                        and not process_unit_main["fixed"]
                    ):
                        continue
                    logger.info(f"{process_unit_main = }")
                    working_time_available_worker_virtual_df = (
                        working_time_available_worker_virtual_df.astype(
                            {"worker_id": float}
                        ).astype({"worker_id": int})
                    )
                    working_time_available_machine_virtual_df = (
                        working_time_available_machine_virtual_df.astype(
                            {"machine_id": float}
                        ).astype({"machine_id": int})
                    )
                    worker_temp_working_df = (
                        working_time_available_worker_virtual_df.copy()
                    )
                    machine_temp_working_df = (
                        working_time_available_machine_virtual_df.copy()
                    )
                    # Pre processing lunch break time
                    if process_unit_main["linker_id"]:
                        list_process_linker = []
                        list_process_linker.append(process_unit_main)
                        linker_id = process_unit_main["linker_id"]
                        while True:
                            process_unit_linker = process_dict_query[linker_id]
                            list_process_linker.append(process_unit_linker)
                            if not process_unit_linker["linker_id"]:
                                break
                            linker_id = process_unit_linker["linker_id"]
                    else:
                        list_process_linker = [process_unit_main]
                    final_decision_dict = {}
                    force_start = None
                    is_linker_range = False
                    if len(list_process_linker) >= 2:
                        is_linker_range = True
                    list_result_schedule = []
                    for idx, process_unit in enumerate(list_process_linker):
                        if idx >= 1:
                            logger.info(f"{process_unit = }")
                        if self.MORNING_END_TIME == self.AFTERNOON_START_TIME:
                            process_unit["is_work_break_time"] = True
                        # Handle fixed processes
                        if process_unit["fixed"]:
                            date_df = working_time_available_worker_virtual_df[
                                working_time_available_worker_virtual_df.process_id
                                == process_unit["process_id"]
                            ]
                            process_schedule[process_unit["process_id"]] = [
                                date_df.start_date.min(),
                                date_df.end_date.max(),
                            ]
                            if process_unit["automatic_work"]:
                                data_machine_df = working_time_available_machine_virtual_df[
                                    working_time_available_machine_virtual_df.process_id
                                    == process_unit["process_id"]
                                ]
                            temp_start_at = datetime.strptime(
                                process_unit["start_at"]
                                .replace("T", " ")
                                .replace("Z", "")
                                .split("+")[0],
                                "%Y-%m-%d %H:%M:%S",
                            )
                            temp_end_at = datetime.strptime(
                                process_unit["end_at"]
                                .replace("T", " ")
                                .replace("Z", "")
                                .split("+")[0],
                                "%Y-%m-%d %H:%M:%S",
                            )
                            work_time = [temp_start_at, temp_end_at]
                            process_schedule[process_unit["process_id"]] = work_time
                            continue
                        # Check ancestor was finished or not
                        dependency = (
                            process_unit["depend_on"]
                            if process_unit["depend_on"] is not None
                            else []
                        )
                        if len(dependency):
                            dependency = process_unit["depend_on"]
                            check = [p for p in dependency if p in process_schedule]
                            if process_unit_main["linker_id"] and idx >= 1:
                                check.append(list_process_linker[idx - 1]["linker_id"])
                            if len(check) == 0:
                                raise RuntimeError(
                                    f"Dependency of {process_unit['process_id']} is not completed"
                                )
                            list_finished_task = [
                                v[1]
                                for k, v in process_schedule.items()
                                if k in dependency
                            ]
                            if list_finished_task:
                                self.dt_current_date_adjusted = max(list_finished_task)
                            self.dt_current_date_adjusted = max(
                                self.dt_current_date_adjusted, self.dt_current_date
                            )
                        else:
                            self.dt_current_date_adjusted = self.dt_current_date
                        if process_unit_main["linker_id"] and idx >= 1:
                            ...
                        else:
                            if len(process_unit["material_deadline"]):
                                try:
                                    material_deadline = datetime.strptime(
                                        process_unit["material_deadline"]
                                        .replace("T", " ")
                                        .replace("Z", "")
                                        .split("+")[0],
                                        "%Y-%m-%d %H:%M:%S",
                                    )
                                except:
                                    material_deadline = datetime.strptime(
                                        process_unit["material_deadline"], "%Y-%m-%d"
                                    )
                                self.dt_current_date_adjusted = max(
                                    self.dt_current_date_adjusted, material_deadline
                                )

                            list_worker_id, list_machine_id, _ = (
                                self.get_factory_resources_list(
                                    process_unit=process_unit,
                                    worker_details_df=worker_details_df,
                                )
                            )
                            self.dict_total_worked_time = {}
                            self.dict_total_worked_time["worker"] = {}
                            self.dict_total_worked_time["machine"] = {}
                            worker_id_df = working_time_available_worker_virtual_df[
                                working_time_available_worker_virtual_df[
                                    "worker_id"
                                ].isin(list_worker_id)
                            ]
                            if len(worker_id_df):
                                total_duration = worker_temp_working_df.drop_duplicates(
                                    subset="process_id", keep="first"
                                )
                                worker_total_duration = total_duration.groupby(
                                    "worker_id"
                                )["processing_time"].sum()
                                self.dict_total_worked_time["worker"] = (
                                    worker_total_duration.to_dict()
                                )

                                worker_id_df["next_start_date"] = worker_id_df[
                                    "start_date"
                                ].shift(-1)
                                worker_id_df["diff_mins"] = (
                                    worker_id_df["next_start_date"]
                                    - worker_id_df["end_date"]
                                ).dt.total_seconds() / 60
                                worker_id_df_tmp = worker_id_df[
                                    worker_id_df["diff_mins"]
                                    > process_unit["machining_time"]
                                ]
                                self.dt_current_date_adjusted = max(
                                    self.dt_current_date_adjusted,
                                    worker_id_df_tmp["end_date"].min(),
                                )

                            machine_id_df = working_time_available_machine_virtual_df[
                                working_time_available_machine_virtual_df[
                                    "machine_id"
                                ].isin(list_machine_id)
                            ]
                            if len(machine_id_df):
                                total_duration = (
                                    machine_temp_working_df.drop_duplicates(
                                        subset="process_id", keep="first"
                                    )
                                )
                                machine_total_duration = total_duration.groupby(
                                    "machine_id"
                                )["processing_time"].sum()
                                self.dict_total_worked_time["machine"] = (
                                    machine_total_duration.to_dict()
                                )

                                machine_id_df["next_start_date"] = machine_id_df[
                                    "start_date"
                                ].shift(-1)
                                machine_id_df["diff_mins"] = (
                                    machine_id_df["next_start_date"]
                                    - machine_id_df["end_date"]
                                ).dt.total_seconds() / 60
                                machine_id_df_tmp = machine_id_df[
                                    machine_id_df["diff_mins"]
                                    > process_unit["machining_time"]
                                ]
                                self.dt_current_date_adjusted = max(
                                    self.dt_current_date_adjusted,
                                    machine_id_df_tmp["end_date"].min(),
                                )
                        if have_same_shape:
                            if process_unit["process_id"] != cluster[0]:
                                self.dt_current_date_adjusted = max(
                                    self.dt_current_date_adjusted,
                                    process_schedule[cluster[0]][1],
                                )

                        max_end_proces = pd.concat(
                            [worker_id_df["end_date"], machine_id_df["end_date"]]
                        ).max()
                        arrange = True
                        if (
                            process_unit["process_id"]
                            in self.process_map_object.DEPEND_DICT
                        ):
                            depend_process = working_time_available_worker_virtual_df[
                                working_time_available_worker_virtual_df.process_id.isin(
                                    self.process_map_object.DEPENT_FORWARD[
                                        process_unit["process_id"]
                                    ]
                                )
                            ]
                            if len(depend_process):
                                self.dt_delivery_date_adjusted = (
                                    depend_process.start_date.min()
                                )
                            else:
                                self.dt_delivery_date_adjusted = datetime.strptime(
                                    process_unit["final_deadline"]
                                    .replace("T", " ")
                                    .replace("Z", "")
                                    .split("+")[0],
                                    "%Y-%m-%d %H:%M:%S",
                                )
                        else:
                            self.dt_delivery_date_adjusted = datetime.strptime(
                                process_unit["final_deadline"]
                                .replace("T", " ")
                                .replace("Z", "")
                                .split("+")[0],
                                "%Y-%m-%d %H:%M:%S",
                            )
                        if not process_unit["machining_time"]:
                            logger.info(f"start time {self.dt_current_date_adjusted}")
                            logger.info(
                                f"estimated deadline {self.dt_delivery_date_adjusted}"
                            )
                            logger.info(
                                f"Plainning process {process_unit['process_id']} no processing time"
                            )
                            work_time = [
                                self.dt_current_date_adjusted,
                                self.dt_current_date_adjusted,
                            ]
                            process_schedule_human_tmp = {
                                "range": work_time,
                                "worker_id": process_unit["worker_id"],
                                "machine_id": process_unit["machine_id"],
                                "worker_time": work_time,
                                "process_time": [work_time],
                            }
                            process_schedule_human[process_unit["process_id"]] = (
                                process_schedule_human_tmp
                            )
                            process_schedule[process_unit["process_id"]] = work_time
                            new_process = pd.DataFrame()
                            new_process["start_date"] = process_unit["start_at"]
                            new_process["end_date"] = process_unit["end_at"]
                            new_process["worker_id"] = process_unit["worker_id"]
                            new_process["process_id"] = process_unit["process_id"]
                            new_process["machine_id"] = process_unit["machine_id"]
                            new_process["processing_time"] = process_unit[
                                "machining_time"
                            ]

                            working_time_available_worker_virtual_df = pd.concat(
                                [working_time_available_worker_virtual_df, new_process]
                            )
                            working_time_available_worker_virtual_df = (
                                working_time_available_worker_virtual_df.sort_values(
                                    "start_date", ascending=True
                                )
                            )

                            working_time_available_machine_virtual_df = pd.concat(
                                [working_time_available_machine_virtual_df, new_process]
                            )
                            working_time_available_machine_virtual_df = (
                                working_time_available_machine_virtual_df.sort_values(
                                    by=["start_date"], ascending=True
                                )
                            )
                            arrange = False
                            if self.batch_return:
                                self.__push_results_to_db(
                                    doc={
                                        "process_schedule": process_schedule_human_tmp,
                                        "process_id": process_unit["process_id"],
                                        "request_id": self.request_id,
                                        "logs": constants.SUCCESS_LOGS,
                                        "status": "in-progress",
                                        "total_card": total_card,
                                        "request_id": self.request_id,
                                    }
                                )
                            continue
                        if not process_unit["is_internal"] and arrange:
                            logger.info(f"start time {self.dt_current_date_adjusted}")
                            logger.info(
                                f"estimated deadline {self.dt_delivery_date_adjusted}"
                            )
                            logger.info(
                                f"Plainning process {process_unit['process_id']} is external"
                            )
                            decision_dict_external = {}
                            if is_linker_range and idx >= 1 and not force_start:
                                for id, info in decision_dict_tmp.items():
                                    worker_id, machine_id, allocated_time_list = (
                                        self.build_allocated_time_list_external(
                                            start=self.max_end_date_in_dict(
                                                info["allocated_time_dict"]
                                            ),
                                            total_processing_time=process_unit[
                                                "machining_time"
                                            ],
                                            is_included_holiday=process_unit[
                                                "is_included_holiday"
                                            ],
                                            worker_id=process_unit["worker_id"],
                                            process_id=process_unit["machine_id"],
                                        )
                                    )
                                    allocated_time_list = [
                                        x for x in allocated_time_list if x[0] != x[1]
                                    ]
                                    decision_dict_external[id] = {
                                        "end": allocated_time_list[-1][1],
                                        "overtime_hours": 0,
                                        "operating_rate": 0,
                                        "is_not_auto": 0,
                                        "allocated_time_dict": {
                                            constants.PROCESSING_STEP: allocated_time_list
                                        },
                                    }
                                final_decision_dict[process_unit["process_id"]] = (
                                    decision_dict_external
                                )
                            else:
                                worker_id = ""
                                machine_id = ""
                                temp_processing_time = process_unit["machining_time"]
                                self.process_map_object.COMPLETE_PROCESS_ID.append(
                                    process_unit["process_id"]
                                )
                                worker_id, machine_id, allocated_time_list = (
                                    self.build_allocated_time_list_external(
                                        start=(
                                            force_start
                                            if force_start
                                            else self.dt_current_date_adjusted
                                        ),
                                        total_processing_time=temp_processing_time,
                                        is_included_holiday=process_unit[
                                            "is_included_holiday"
                                        ],
                                        worker_id=process_unit["worker_id"],
                                        process_id=process_unit["machine_id"],
                                    )
                                )
                                allocated_time_list = [
                                    x for x in allocated_time_list if x[0] != x[1]
                                ]
                                process_schedule_human_tmp = {
                                    "range": [
                                        allocated_time_list[0][0],
                                        allocated_time_list[-1][1],
                                    ],
                                    "worker_id": process_unit["worker_id"],
                                    "machine_id": process_unit["machine_id"],
                                    "worker_time": allocated_time_list,
                                    "process_time": allocated_time_list,
                                    "is_overtime_expected": False,
                                    "is_nightshift_expected": False,
                                }
                                process_schedule_human[process_unit["process_id"]] = (
                                    process_schedule_human_tmp
                                )
                                if self.batch_return:
                                    self.__push_results_to_db(
                                        doc={
                                            "process_schedule": process_schedule_human_tmp,
                                            "process_id": process_unit["process_id"],
                                            "request_id": self.request_id,
                                            "logs": constants.SUCCESS_LOGS,
                                            "status": "in-progress",
                                            "total_card": total_card,
                                            "request_id": self.request_id,
                                        }
                                    )

                                process_schedule[process_unit["process_id"]] = [
                                    allocated_time_list[0][0],
                                    allocated_time_list[-1][1],
                                ]

                                new_process = pd.DataFrame()
                                new_process["start_date"] = [
                                    x[0] for x in allocated_time_list
                                ]
                                new_process["end_date"] = [
                                    x[1] for x in allocated_time_list
                                ]
                                new_process["worker_id"] = process_unit["worker_id"]
                                new_process["machine_id"] = process_unit["machine_id"]
                                new_process["process_id"] = process_unit["process_id"]
                                new_process["processing_time"] = temp_processing_time
                                working_time_available_worker_virtual_df = pd.concat(
                                    [
                                        working_time_available_worker_virtual_df,
                                        new_process,
                                    ]
                                )
                                working_time_available_worker_virtual_df = working_time_available_worker_virtual_df.sort_values(
                                    "start_date", ascending=True
                                )

                                working_time_available_machine_virtual_df = pd.concat(
                                    [
                                        working_time_available_machine_virtual_df,
                                        new_process,
                                    ]
                                )
                                working_time_available_machine_virtual_df = working_time_available_machine_virtual_df.sort_values(
                                    by=["start_date"], ascending=True
                                )
                                arrange = False
                                force_start = allocated_time_list[-1][1]
                                continue
                        worker_id = ""
                        is_overtime = process_unit["overtime"]
                        list_stage = []
                        if (
                            self.dt_delivery_date_adjusted
                            <= self.dt_current_date_adjusted
                        ):
                            list_stage.append(
                                {
                                    "deadline": (
                                        max_end_proces
                                        if not pd.isnull(max_end_proces)
                                        and max_end_proces
                                        > self.dt_current_date_adjusted
                                        else self.dt_current_date_adjusted
                                    ),
                                    "keep_deadline": False,
                                }
                            )
                        else:
                            list_stage.append(
                                {
                                    "deadline": self.dt_delivery_date_adjusted,
                                    "keep_deadline": True,
                                }
                            )
                            list_stage.append(
                                {
                                    "deadline": (
                                        max_end_proces
                                        if not pd.isnull(max_end_proces)
                                        and max_end_proces
                                        > self.dt_delivery_date_adjusted
                                        else self.dt_delivery_date_adjusted
                                    ),
                                    "keep_deadline": False,
                                }
                            )
                        common_resource_args = dict(
                            process_unit=process_unit,
                            worker_details_virtual_df=worker_details_virtual_df,
                            working_time_available_worker_df=worker_temp_working_df,
                            machine_details_virtual_df=machine_details_virtual_df,
                            working_time_available_machine_df=machine_temp_working_df,
                        )
                        if force_start:
                            decision_dict_tmp = {}
                        for stage in list_stage:
                            logger.info(f"start time {self.dt_current_date_adjusted}")
                            logger.info(f'estimated deadline {stage["deadline"]}')
                            if stage["keep_deadline"]:
                                if (
                                    process_unit["is_nightshift_expected"]
                                    and self.settings.nightshift_expected
                                ):
                                    logger.info(
                                        f"Process {process_unit['process_id']}, checking nightshift{', overtime' if process_unit['overtime'] else ''} with night_time_prioritize"
                                    )
                                    with self.adjust_schedule_time(
                                        overtime=process_unit["overtime"]
                                    ):
                                        for start, end in self.list_breaktime:
                                            if start <= self.WORKING_HOURS_END <= end:
                                                self.WORKING_HOURS_END = start
                                        (
                                            avail_time,
                                            machine_avail_time,
                                            dict_worker_matching,
                                        ) = self.get_reasonable_resources(
                                            **common_resource_args,
                                            current_date=self.dt_current_date_adjusted,
                                            delivery_date=stage["deadline"],
                                        )
                                        avail_time_working_nightshift_expected = {}
                                        avail_time_working_tmp = avail_time
                                        for k, v in avail_time_working_tmp.items():
                                            worker_id_tmp, machine_id_tmp = k.split(
                                                "___"
                                            )
                                            if (
                                                int(worker_id_tmp)
                                                in dict_worker_matching.get(
                                                    int(machine_id_tmp), []
                                                )
                                                or dict_worker_matching == {}
                                            ):
                                                if self.factory_rs_info[
                                                    "dict_machine_auto"
                                                ].get(int(machine_id_tmp), False):
                                                    avail_time_working_nightshift_expected[
                                                        k
                                                    ] = v
                                        if avail_time_working_nightshift_expected:
                                            if process_unit_main["linker_id"]:
                                                decision_dict_tmp = self.schedule_core_linker(
                                                    process_unit=process_unit,
                                                    decision_dict_child=(
                                                        decision_dict_tmp
                                                        if idx >= 1
                                                        else {}
                                                    ),
                                                    factory_resources=avail_time,
                                                    deadline=stage["deadline"],
                                                    overtime=process_unit["overtime"],
                                                    time_grouper_machine=machine_avail_time,
                                                    force_start=force_start,
                                                )
                                            else:
                                                list_result_schedule = self.schedule_core(
                                                    process_unit=process_unit,
                                                    factory_resources=avail_time_working_nightshift_expected,
                                                    deadline=stage["deadline"],
                                                    night_time_prioritize=True,
                                                    time_grouper_machine=machine_avail_time,
                                                )
                                if len(worker_id) == 0:
                                    logger.info(
                                        f"Process {process_unit['process_id']}, checking with default card configs"
                                    )
                                    with self.adjust_schedule_time(
                                        overtime=process_unit["overtime"]
                                    ):
                                        for start, end in self.list_breaktime:
                                            if start <= self.WORKING_HOURS_END <= end:
                                                self.WORKING_HOURS_END = start
                                        (
                                            avail_time,
                                            machine_avail_time,
                                            dict_worker_matching,
                                        ) = self.get_reasonable_resources(
                                            **common_resource_args,
                                            current_date=self.dt_current_date_adjusted,
                                            delivery_date=stage["deadline"],
                                        )
                                        if process_unit_main["linker_id"]:
                                            decision_dict_tmp = self.schedule_core_linker(
                                                process_unit=process_unit,
                                                decision_dict_child=(
                                                    decision_dict_tmp
                                                    if idx >= 1
                                                    else {}
                                                ),
                                                factory_resources=avail_time,
                                                deadline=stage["deadline"],
                                                overtime=process_unit["overtime"],
                                                time_grouper_machine=machine_avail_time,
                                                force_start=force_start,
                                            )
                                        else:
                                            list_result_schedule = self.schedule_core(
                                                process_unit=process_unit,
                                                factory_resources=avail_time,
                                                deadline=stage["deadline"],
                                                overtime=process_unit["overtime"],
                                                time_grouper_machine=machine_avail_time,
                                            )
                            else:
                                logger.info(
                                    f"Process {process_unit['process_id']}, deadlines don't fit, find the best possible schedule"
                                )
                                is_overtime = max(
                                    self.settings.overtime_expected,
                                    process_unit["overtime"],
                                )
                                with self.adjust_schedule_time(overtime=is_overtime):
                                    deadline_adjust = stage["deadline"] + timedelta(
                                        days=1
                                    )
                                    while self.is_day_off(deadline_adjust):
                                        deadline_adjust += timedelta(days=1)
                                    deadline_adjust = datetime.combine(
                                        deadline_adjust.date(), self.WORKING_HOURS_END
                                    )
                                    logger.info(
                                        f"start time {self.dt_current_date_adjusted}"
                                    )
                                    logger.info(f"estimated deadline {deadline_adjust}")
                                    for start, end in self.list_breaktime:
                                        if start <= self.WORKING_HOURS_END <= end:
                                            self.WORKING_HOURS_END = start
                                    (
                                        avail_time,
                                        machine_avail_time,
                                        dict_worker_matching,
                                    ) = self.get_reasonable_resources(
                                        **common_resource_args,
                                        current_date=self.dt_current_date_adjusted,
                                        delivery_date=deadline_adjust,
                                    )
                                    if process_unit_main["linker_id"]:
                                        decision_dict_tmp = self.schedule_core_linker(
                                            process_unit=process_unit,
                                            decision_dict_child=(
                                                decision_dict_tmp if idx >= 1 else {}
                                            ),
                                            factory_resources=avail_time,
                                            deadline=deadline_adjust,
                                            overtime=process_unit["overtime"],
                                            time_grouper_machine=machine_avail_time,
                                            force_start=force_start,
                                        )
                                    else:
                                        list_result_schedule = self.schedule_core(
                                            process_unit=process_unit,
                                            factory_resources=avail_time,
                                            deadline=deadline_adjust,
                                            overtime=is_overtime,
                                            keep_deadline=False,
                                            time_grouper_machine=machine_avail_time,
                                        )
                            if process_unit_main["linker_id"]:
                                if decision_dict_tmp:
                                    final_decision_dict[process_unit["process_id"]] = (
                                        decision_dict_tmp
                                    )
                                    force_start = None
                                    break
                            else:
                                if len(list_result_schedule[0][0]):
                                    break
                    if is_linker_range:
                        ...
                    else:
                        if (
                            not process_unit_main["is_internal"]
                            or not process_unit_main["machining_time"]
                            or process_unit_main["fixed"]
                        ):
                            continue
                    if process_unit_main["linker_id"]:
                        for k, v in final_decision_dict.items():
                            chains = self.build_chains(v)
                            ids = chains[0][0]
                            (
                                worker_id,
                                machine_id,
                                _,
                            ) = ids.split("___")
                            allocated_time_dict = final_decision_dict[k][ids][
                                "allocated_time_dict"
                            ]
                            list_result_schedule.append(
                                (worker_id, machine_id, k, allocated_time_dict)
                            )
                    if len(list_result_schedule):
                        logger.info("*" * 80)
                        for (
                            worker_id,
                            machine_id,
                            process_id,
                            allocated_time_dict,
                        ) in list_result_schedule:
                            if not worker_id:
                                raise RuntimeError("Something error")
                            process_unit = process_dict_query[process_id]
                            if not process_unit["is_internal"]:
                                worker_id = process_unit["worker_id"]
                                machine_id = process_unit["machine_id"]
                            logger.info(f"{worker_id = }")
                            logger.info(f"{machine_id = }")
                            logger.info(f"{process_id = }")
                            worker_allocated_time_list = [
                                v for _, v in allocated_time_dict.items()
                            ]
                            worker_allocated_time_list = [
                                item
                                for sublist in worker_allocated_time_list
                                for item in sublist
                            ]
                            worker_allocated_time_list = [
                                x for x in worker_allocated_time_list if x[0] != x[1]
                            ]
                            self.process_map_object.COMPLETE_PROCESS_ID.append(
                                process_id
                            )
                            temp_processing_time = process_unit["machining_time"]
                            process_schedule[process_id] = [
                                worker_allocated_time_list[0][0],
                                worker_allocated_time_list[-1][1],
                            ]
                            process_schedule_human_tmp = {
                                "range": [
                                    worker_allocated_time_list[0][0],
                                    worker_allocated_time_list[-1][1],
                                ],
                                "worker_id": worker_id,
                                "machine_id": machine_id,
                                "worker_time": worker_allocated_time_list,
                                "process_time": worker_allocated_time_list,
                                "is_overtime_expected": is_overtime,
                                "is_nightshift_expected": process_unit[
                                    "is_nightshift_expected"
                                ],
                            }
                            process_schedule_human[process_id] = (
                                process_schedule_human_tmp
                            )
                            new_process = pd.DataFrame()
                            new_process["start_date"] = [
                                x[0] for x in worker_allocated_time_list
                            ]
                            new_process["end_date"] = [
                                x[1] for x in worker_allocated_time_list
                            ]
                            new_process["worker_id"] = worker_id
                            new_process["machine_id"] = machine_id
                            new_process["process_id"] = process_id
                            new_process["processing_time"] = temp_processing_time
                            working_time_available_machine_virtual_df = pd.concat(
                                [working_time_available_machine_virtual_df, new_process]
                            )
                            working_time_available_machine_virtual_df = (
                                working_time_available_machine_virtual_df.sort_values(
                                    by=["process_id", "start_date"], ascending=True
                                )
                            )
                            if self.factory_rs_info["dict_machine_auto"].get(
                                int(machine_id), False
                            ):
                                temp_processing_time = process_unit["machining_time"]
                                worker_allocated_time_list = [
                                    v
                                    for k, v in allocated_time_dict.items()
                                    if k != "processing"
                                ]
                                worker_allocated_time_list = [
                                    item
                                    for sublist in worker_allocated_time_list
                                    for item in sublist
                                ]
                                worker_allocated_time_list = [
                                    x
                                    for x in worker_allocated_time_list
                                    if x[0] != x[1]
                                ]
                                new_process = pd.DataFrame()
                                new_process["start_date"] = [
                                    x[0] for x in worker_allocated_time_list
                                ]
                                new_process["end_date"] = [
                                    x[1] for x in worker_allocated_time_list
                                ]
                                new_process["worker_id"] = worker_id
                                new_process["process_id"] = process_unit["process_id"]
                                new_process["automatic_work"] = True
                                new_process["nightshift_expected"] = process_unit[
                                    "is_nightshift_expected"
                                ]
                                new_process["processing_time"] = temp_processing_time
                                working_time_available_worker_virtual_df = pd.concat(
                                    [
                                        working_time_available_worker_virtual_df,
                                        new_process,
                                    ]
                                )
                                working_time_available_worker_virtual_df = working_time_available_worker_virtual_df.sort_values(
                                    by=["process_id", "start_date"], ascending=True
                                )
                            else:
                                working_time_available_worker_virtual_df = pd.concat(
                                    [
                                        working_time_available_worker_virtual_df,
                                        new_process,
                                    ]
                                )
                                working_time_available_worker_virtual_df = working_time_available_worker_virtual_df.sort_values(
                                    by=["start_date"], ascending=True
                                )
                            logger.info("*" * 80)
                            if self.batch_return:
                                self.__push_results_to_db(
                                    doc={
                                        "process_schedule": process_schedule_human_tmp,
                                        "process_id": process_unit["process_id"],
                                        "request_id": self.request_id,
                                        "logs": constants.SUCCESS_LOGS,
                                        "status": "in-progress",
                                        "total_card": total_card,
                                        "request_id": self.request_id,
                                    }
                                )
                    else:
                        logs = "Fujin failed at optimization stage"
                        raise RuntimeError(logs)
        # logger.info(f'{process_schedule_human = }')
        return process_schedule_human
