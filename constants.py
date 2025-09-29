class Constants:
    TITLE: str = "Fujin API"
    VERSION: str = "0.16.0"
    DT_FORMAT = "%Y-%m-%d %H:%M:%S"
    WEEKDAY_MAP = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    FIXED_STATUS = (4, 21, 22, 23)
    PRE_PROCESS_STEP = "pre_processing"
    PROCESSING_STEP = "processing"
    POST_PROCESS_STEP = "post_processing"

    PREVIOUS_PROCESS_STEP = {
        PRE_PROCESS_STEP: None,
        PROCESSING_STEP: PRE_PROCESS_STEP,
        POST_PROCESS_STEP: PROCESSING_STEP,
    }

    ESLATICSEARCH_RESULTS_TASK_LIST_INDICE = "fujin-results-task-list-space"
    ESLATICSEARCH_STATUS_JOB_ID = "fujin-job-status-space"
    SUCCESS_LOGS = "complete"

    MYSQL_COMMAND_TAKE_SHAPE = """
        SELECT 
            pl.id AS production_lot_id,
            pl.physical_type_id,
            pl.base_id,
            do.result
        FROM production.production_lots pl
        LEFT JOIN physical_obj.diagram_ocr do 
            ON pl.physical_type_id = do.physical_type_id
        WHERE do.result IS NOT NULL
        AND pl.organization_id = %s
        AND pl.base_id = %s
    """

    OTHER_PRODUCT_SHAPE_NAME = {"shape": "OTHERS", "dimension": None}


constants = Constants()
