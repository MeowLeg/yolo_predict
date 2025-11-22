import sqlite3

from yolo_predict.utils.detect import Image


def read_images_from_db(cur: sqlite3.Cursor, limit: int) -> list[Image]:
    # 从数据库读取图片进行检测
    # 直接pic表left join下stream_tag表
    rs: list[sqlite3.Row] = cur.execute(
        """
        select id, path, uuid, ifnull(code, '') as code
        from pic
        left join stream_tag
            on pic.uuid = stream_tag.uuid
        where predicted = 0
        limit ?
    """,
        (limit,),
    ).fetchall()

    # 确保同图片组不被切割成2次批处理，可能需要测试性能
    last_image_id = rs[-1]["id"]
    while True:
        r: sqlite3.Row = cur.execute(
            """
            select id, path, uuid, ifnull(code, '') as code
            from pic
            left join stream_tag
                on pic.uuid = stream_tag.uuid
            where predicted = 0 and id > ?
            limit 1
            """,
            (last_image_id,),
        ).fetchone()
        if r is None or r["id"] != last_image_id:
            break
        else:
            rs.append(r)

    images = [Image(**dict(r)) for r in rs]
    return images
