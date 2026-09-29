import json
import posixpath
import re
import zipfile
from xml.etree import ElementTree


class TaskInputError(ValueError):
    pass


def parse_task_file(upload) -> list[dict[str, str]]:
    name = (getattr(upload, "name", "") or "").lower()
    try:
        raw = upload.read()
        if name.endswith(".json"):
            value = json.loads(raw.decode("utf-8-sig"))
        elif name.endswith(".xlsx"):
            value = _read_xlsx(raw)
        else:
            raise TaskInputError("Only .json and .xlsx files are supported.")
    except TaskInputError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise TaskInputError("The task file is not a valid JSON or XLSX file.") from exc
    return _normalize(value)


def _normalize(value) -> list[dict[str, str]]:
    if isinstance(value, dict):
        value = value.get("tasks", value.get("objectives", value.get("items", value)))
    if not isinstance(value, list):
        raise TaskInputError("The task file must contain an array of task objectives.")

    result = []
    for item in value:
        if isinstance(item, str):
            objective = item.strip()
        elif isinstance(item, dict):
            objective = next((item.get(key) for key in ("objective", "task", "target", "description") if item.get(key)), "")
            objective = str(objective).strip()
        else:
            continue
        if objective:
            result.append({"objective": objective})
    if not result:
        raise TaskInputError("The task file does not contain any non-empty objectives.")
    return result


def _read_xlsx(raw: bytes):
    with zipfile.ZipFile(__import__("io").BytesIO(raw)) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root]
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheet = workbook.find(".//{*}sheet")
        if sheet is None:
            raise TaskInputError("The XLSX file has no worksheet.")
        target = rel_map[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
        path = posixpath.normpath(posixpath.join("xl", target))
        root = ElementTree.fromstring(archive.read(path))
        rows = []
        for row in root.findall(".//{*}row"):
            cells = []
            for cell in row.findall("{*}c"):
                value = cell.find("{*}v")
                text = "" if value is None else value.text or ""
                if cell.attrib.get("t") == "s" and text:
                    text = shared[int(text)]
                cells.append(text)
            if cells:
                rows.append(cells)
        if not rows:
            return []
        headers = [re.sub(r"\s+", "", str(v).lower()) for v in rows[0]]
        index = next((headers.index(key) for key in ("objective", "task", "target", "description", "任務目標") if key in headers), 0)
        return [{"objective": row[index] if index < len(row) else ""} for row in rows[1:]]
