import pymysql
from settings import settings

# Connect to your production database
conn = pymysql.connect(
    host=settings.PRODUCTION_DATABASE_HOST,
    user=settings.PRODUCTION_DATABASE_USER,
    password=settings.PRODUCTION_DATABASE_PASSWORD,
    database=settings.PRODUCTION_DATABASE_DATABASE,
    cursorclass=pymysql.cursors.DictCursor,
    port=getattr(settings, "PRODUCTION_DATABASE_PORT", 3306),
    connect_timeout=5,
)

try:
    with conn.cursor() as cursor:
        # --- First query: tool_nos ---
        tool_query = """
        SELECT 
            tool_nos.id AS tool_no_id, 
            tool_nos.name AS tool_no_name
        FROM tool_nos
        JOIN planning_process_tool_nos pptn ON tool_nos.id = pptn.tool_no_id
        JOIN planning_processes pp ON pptn.planning_process_id = pp.id
        JOIN production_lots pl ON pp.production_lot_id = pl.id
        WHERE pl.id = %s;
        """
        cursor.execute(tool_query, (5975,))
        tool_nos = cursor.fetchall()

        # --- Second query: jigs ---
        jig_query = """
        SELECT 
            jigs.id AS jig_id, 
            jigs.name AS jig_name
        FROM jigs
        JOIN planning_process_jigs ppj ON jigs.id = ppj.jig_id
        JOIN planning_processes pp ON ppj.planning_process_id = pp.id
        JOIN production_lots pl ON pp.production_lot_id = pl.id
        WHERE pl.id = %s;
        """
        cursor.execute(jig_query, (5856,))
        jigs = cursor.fetchall()

    print("=== TOOL NUMBERS ===")
    for t in tool_nos:
        print(f"ID: {t['tool_no_id']}, Name: {t['tool_no_name']}")

    print("\n=== JIGS ===")
    for j in jigs:
        print(f"ID: {j['jig_id']}, Name: {j['jig_name']}")

finally:
    conn.close()
