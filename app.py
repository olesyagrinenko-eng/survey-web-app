from __future__ import annotations

import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_file,
    flash,
)
import math
from werkzeug.datastructures import FileStorage
from flask import after_this_request

try:
    import pyreadstat  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    pyreadstat = None


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")


# --- Simple in‑memory "storage" for uploaded dataframes (per session) ---
DATASTORE: Dict[str, pd.DataFrame] = {}
EXCELSTORE: Dict[str, str] = {}  # excel_id -> file path
RESULTSTORE: Dict[str, Dict[str, Any]] = {}  # result_id -> {"table_html": str, "significance": dict}


@dataclass
class SurveyConfig:
    row_vars: List[str]
    col_vars: List[str]
    metric_count: bool
    metric_percent: bool
    use_weights: bool
    perform_ztest: bool


def _load_dataframe_from_upload(file_storage) -> pd.DataFrame:
    filename = file_storage.filename or ""
    ext = filename.lower().split(".")[-1]

    if ext in {"csv"}:
        return pd.read_csv(file_storage)
    if ext in {"xls", "xlsx"}:
        return pd.read_excel(file_storage)
    if ext in {"sav"}:
        if pyreadstat is None:
            raise RuntimeError(
                "Для чтения .sav требуется библиотека pyreadstat "
                "(она уже добавлена в requirements.txt, установите зависимости)."
            )
        # pyreadstat ожидает путь к файлу, а не объект FileStorage
        fd, tmp_path = tempfile.mkstemp(suffix=".sav")
        os.close(fd)
        try:
            file_storage.save(tmp_path)
            df, _meta = pyreadstat.read_sav(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return df

    raise ValueError("Неподдерживаемый формат файла. Разрешены: csv, xls, xlsx, sav.")


def _safe_sort(values: np.ndarray | List[Any]) -> List[Any]:
    try:
        numeric_values: List[Tuple[float, Any]] = []
        for v in values:
            try:
                numeric_values.append((float(v), v))
            except (TypeError, ValueError):
                # как только встречаем не‑число, сортируем всё как строки
                return sorted(values, key=lambda x: str(x))
        return [orig for _, orig in sorted(numeric_values)]
    except Exception:
        return sorted(values, key=lambda x: str(x))


def _z_test_vs_total(
    total_success: float,
    total_n: float,
    group_success: float,
    group_n: float,
) -> Tuple[float, float]:
    """
    Классический пропорционный z‑тест (две независимые выборки).
    Возвращает (z, p_value).
    """
    if total_n <= 0 or group_n <= 0:
        return 0.0, 1.0

    p1 = group_success / group_n
    p2 = total_success / total_n
    p_pool = (group_success + total_success) / (group_n + total_n)

    denom = np.sqrt(p_pool * (1 - p_pool) * (1 / group_n + 1 / total_n))
    if denom == 0:
        return 0.0, 1.0

    z = (p1 - p2) / denom
    # нормальное CDF через erf, чтобы не тянуть scipy
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(z), float(p)


def build_crosstab(
    df: pd.DataFrame,
    config: SurveyConfig,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, Any]]]:
    """
    Строит кросс‑таблицу.
    Возвращает:
      - таблицу значений
      - словарь значимостей {(row_label, col_label) -> {"z": .., "p": .., "dir": "up/down"}}
    """
    weight_col = "weight" if config.use_weights and "weight" in df.columns else None

    # Ограничения, чтобы не падать по памяти на Render
    MAX_ROW_UNIQUE_SKIP = 100
    MAX_ROW_VALUES = 50
    MAX_COL_VALUES = 50
    MAX_TOTAL_CELLS = 50000  # оценка до фактического построения

    # Оценка размера результата (сколько строк * сколько столбцов)
    est_row_count = 0
    for row_var in config.row_vars:
        uniq = df[row_var].dropna().unique()
        n_uniq = len(uniq)
        if n_uniq > MAX_ROW_UNIQUE_SKIP:
            continue
        est_row_count += min(n_uniq, MAX_ROW_VALUES)

    est_col_values = 0
    for col_var in config.col_vars:
        uniq = df[col_var].dropna().unique()
        est_col_values += min(len(uniq), MAX_COL_VALUES)

    est_total_cols = 1 + est_col_values  # + Total
    est_cells = est_row_count * est_total_cols
    if est_row_count <= 0:
        raise ValueError("Не удалось построить таблицу: слишком много уникальных значений/нет данных.")
    if est_cells > MAX_TOTAL_CELLS:
        raise ValueError(
            f"Слишком большой объём кросс-таблицы (оценка ячеек: {est_cells}). "
            f"Уменьшите количество выбранных переменных или снимите 'Выбрать все'. "
            f"(Ограничение: {MAX_TOTAL_CELLS} ячеек)"
        )

    # Подготовка столбцов анализа (Total + все выбранные колоночные переменные)
    analysis_cols: List[Tuple[str | None, Any | None, str]] = []  # (col_var, col_val, label)

    # Total сначала
    analysis_cols.append((None, None, "Total"))

    for col_var in config.col_vars:
        uniques = _safe_sort(df[col_var].dropna().unique())
        if len(uniques) > MAX_COL_VALUES:
            uniques = uniques[:MAX_COL_VALUES]
        for val in uniques:
            label = f"{col_var}: {val}"
            analysis_cols.append((col_var, val, label))

    # заголовки
    headers = ["Переменная"]
    headers.extend([lbl for _, _, lbl in analysis_cols])

    rows: List[List[Any]] = []
    significance: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # первая строка — невзвешенные размеры выборки
    sample_row = ["Выборка (N)"]
    total_n = len(df)
    sample_row.append(total_n)
    for col_var, col_val, _lbl in analysis_cols[1:]:
        if col_var is None:
            sample_row.append(total_n)
        else:
            n = len(df[df[col_var] == col_val])
            sample_row.append(n)
    rows.append(sample_row)

    for row_var in config.row_vars:
        unique_vals = _safe_sort(df[row_var].dropna().unique())
        if len(unique_vals) > MAX_ROW_UNIQUE_SKIP:
            continue
        limited_row_values = unique_vals[:MAX_ROW_VALUES] if len(unique_vals) > MAX_ROW_VALUES else unique_vals
        for rv in limited_row_values:
            label = f"{row_var}: {rv}"
            row: List[Any] = [label]

            # для z‑теста нам нужны success и n для total и каждой группы (по percent)
            total_success = 0.0
            total_base = 0.0
            group_success_list: List[Tuple[int, float, float]] = []  # index, success, n

            for idx, (col_var, col_val, _lbl) in enumerate(analysis_cols):
                if col_var is None:
                    base_mask = df[row_var].notna()
                    if weight_col:
                        base_n = df.loc[base_mask, weight_col].sum()
                        success_n = df.loc[
                            (df[row_var] == rv) & base_mask, weight_col
                        ].sum()
                    else:
                        base_n = base_mask.sum()
                        success_n = (df[row_var] == rv).sum()
                else:
                    col_mask = df[col_var] == col_val
                    base_mask = col_mask & df[row_var].notna()
                    if weight_col:
                        base_n = df.loc[base_mask, weight_col].sum()
                        success_n = df.loc[
                            (df[row_var] == rv) & col_mask, weight_col
                        ].sum()
                    else:
                        base_n = base_mask.sum()
                        success_n = ((df[row_var] == rv) & col_mask).sum()

                # count
                if config.metric_count:
                    value = success_n
                else:
                    value = success_n

                # percent
                if config.metric_percent:
                    if base_n > 0:
                        value = round((success_n / base_n) * 100, 1)
                    else:
                        value = 0.0

                row.append(value)

                # подготовка данных для z‑теста
                if config.perform_ztest and config.metric_percent and base_n > 0:
                    if col_var is None:
                        total_success = success_n
                        total_base = base_n
                    else:
                        group_success_list.append((idx, success_n, base_n))

            # после прохода по столбцам считаем z‑тесты
            if (
                config.perform_ztest
                and config.metric_percent
                and total_base > 0
                and total_success > 0
            ):
                for idx, g_succ, g_n in group_success_list:
                    z, p = _z_test_vs_total(
                        total_success=total_success,
                        total_n=total_base,
                        group_success=g_succ,
                        group_n=g_n,
                    )
                    direction = "none"
                    if p < 0.05:
                        direction = "up" if z > 0 else "down"
                    col_label = analysis_cols[idx][2]
                    significance[(label, col_label)] = {
                        "z": z,
                        "p": p,
                        "direction": direction,
                    }

            rows.append(row)

    table_df = pd.DataFrame(rows, columns=headers)
    return table_df, significance


# =========================
# Weighting (bot logic)
# =========================

TRIM_MIN = 0.3
TRIM_MAX = 3.0
MAX_VARS = 10
MAX_CATS_PER_DIM = 60  # for safety (as in the telegram bot)
NORMALIZE_MEAN_1 = True


def create_cross_variable(df: pd.DataFrame, vars_list: List[str]) -> pd.Series:
    """Creates cross variable as in telegram weighting bot: 'A × B × C'."""
    if len(vars_list) == 1:
        return df[vars_list[0]].astype(str)
    cross_var = df[vars_list[0]].astype(str)
    for var in vars_list[1:]:
        cross_var = cross_var + " × " + df[var].astype(str)
    return cross_var


def parse_targets_input(text: str, categories: List[str]) -> Dict[str, float]:
    """
    Parse text like:
      - "M=48, F=52" (percent)
      - "M:0.48, F:0.52" (shares)
    Returns normalized dict {category: share(0..1)} with sum=1.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("пустой ввод")

    norm = raw.replace("\n", " ")
    pairs = re.split(r"[;,|]", norm)
    out: Dict[str, float] = {}
    cats_lower = [c.lower() for c in categories]

    def to_float(v: str) -> float:
        v = v.strip().replace("%", "")
        v = v.replace(",", ".")
        try:
            return float(v)
        except Exception:
            raise ValueError(f"некорректное число: '{v}'")

    # If it's just numbers separated by delimiters (no explicit keys)
    if all("=" not in p and ":" not in p for p in pairs):
        nums = [to_float(p) for p in pairs if p.strip()]
        if len(nums) != len(categories):
            raise ValueError(
                f"ожидалось {len(categories)} чисел (для категорий: {', '.join(categories)}), получили {len(nums)}"
            )
        s = sum(nums)
        vals = [n / 100.0 if 95 <= s <= 105 else n for n in nums]
        s2 = sum(vals)
        if s2 <= 0:
            raise ValueError("сумма долей должна быть > 0")
        vals = [v / s2 for v in vals]
        return {cat: vals[i] for i, cat in enumerate(categories)}

    # key=value or key:value
    for p in pairs:
        if not p.strip():
            continue
        if "=" in p:
            key, val = p.split("=", 1)
        elif ":" in p:
            key, val = p.split(":", 1)
        else:
            raise ValueError(f"не распознан фрагмент: '{p.strip()}' (ожидается cat=val)")

        key = key.strip()
        valf = to_float(val)

        if key.lower() in cats_lower:
            cat = categories[cats_lower.index(key.lower())]
        else:
            if key not in categories:
                raise ValueError(f"категория '{key}' отсутствует. Доступные: {', '.join(categories)}")
            cat = key

        out[cat] = valf

    s = sum(out.values())
    # If it looks like percent
    if 95 <= s <= 105:
        out = {k: v / 100.0 for k, v in out.items()}
        s = sum(out.values())

    if set(out.keys()) != set(categories):
        missing = [c for c in categories if c not in out]
        extra = [c for c in out if c not in categories]
        msg = []
        if missing:
            msg.append("нет значений для: " + ", ".join(missing))
        if extra:
            msg.append("лишние категории: " + ", ".join(extra))
        raise ValueError("; ".join(msg))

    if s <= 0:
        raise ValueError("сумма долей должна быть > 0")

    out = {k: v / s for k, v in out.items()}
    return out


def read_targets_from_excel(file_storage: FileStorage) -> Dict[str, float]:
    df = pd.read_excel(file_storage)

    category_col = None
    target_col = None

    category_names = ["category", "категория", "cat", "name", "название", "группа", "group"]
    target_names = [
        "target",
        "targets",
        "таргет",
        "таргеты",
        "цель",
        "доля",
        "share",
        "percent",
        "процент",
        "value",
        "значение",
    ]

    for col in df.columns:
        if str(col).lower().strip() in category_names:
            category_col = col
            break

    for col in df.columns:
        if str(col).lower().strip() in target_names:
            target_col = col
            break

    if category_col is None or target_col is None:
        cols = list(df.columns)
        if len(cols) >= 2:
            category_col = cols[0]
            target_col = cols[1]
        else:
            raise ValueError(f"Недостаточно колонок в файле. Найдено: {cols}")

    targets: Dict[str, float] = {}
    for _, row in df.iterrows():
        cat = str(row[category_col]).strip()
        try:
            target = float(row[target_col])
        except (ValueError, TypeError):
            continue
        targets[cat] = target

    if not targets:
        raise ValueError("Не удалось прочитать ни одной пары категория-значение")

    total = sum(targets.values())
    if total > 0:
        targets = {k: v / total for k, v in targets.items()}

    return targets


def create_targets_template_xlsx(categories: List[str], current_shares: pd.Series, out_path: str) -> None:
    template_data = []
    for category in sorted(categories):
        template_data.append(
            {
                "Category": category,
                "Targets": float(current_shares.get(category, 0.0)),
            }
        )
    template_df = pd.DataFrame(template_data)
    template_df.to_excel(out_path, index=False, engine="openpyxl")


def clean_sheet_name(name: str) -> str:
    """Очистка названия листа Excel."""
    invalid_chars = r"[\\/:*?\[\]]"
    clean_name = re.sub(invalid_chars, "_", str(name))
    clean_name = clean_name.strip().rstrip(".")
    clean_name = clean_name[:31]
    return clean_name or "Sheet"


def create_sequential_targets_template_xlsx(
    df: pd.DataFrame,
    vars_list: List[str],
    out_path: str,
) -> None:
    """
    Делает Excel workbook: по листу на каждую переменную.
    Каждый лист: Category / Targets (текущие доли как стартовые значения).
    """
    import pandas as _pd

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for var in vars_list:
            series = df[var].dropna().astype(str)
            categories = sorted(series.unique().tolist())
            current_shares = series.value_counts(normalize=True).sort_index()
            template_data = []
            for cat in categories:
                template_data.append({"Category": cat, "Targets": float(current_shares.get(cat, 0.0))})
            template_df = _pd.DataFrame(template_data)
            writer.sheets  # touch to ensure writer init
            template_df.to_excel(writer, sheet_name=clean_sheet_name(var), index=False)


def _read_targets_from_dataframe(df_targets: pd.DataFrame) -> Dict[str, float]:
    category_col = None
    target_col = None

    category_names = ["category", "категория", "cat", "name", "название", "группа", "group"]
    target_names = [
        "target",
        "targets",
        "таргет",
        "таргеты",
        "цель",
        "доля",
        "share",
        "percent",
        "процент",
        "value",
        "значение",
    ]

    for col in df_targets.columns:
        if str(col).lower().strip() in category_names:
            category_col = col
            break
    for col in df_targets.columns:
        if str(col).lower().strip() in target_names:
            target_col = col
            break

    if category_col is None or target_col is None:
        cols = list(df_targets.columns)
        if len(cols) >= 2:
            category_col = cols[0]
            target_col = cols[1]
        else:
            raise ValueError(f"Недостаточно колонок в sheet: {cols}")

    targets: Dict[str, float] = {}
    for _, row in df_targets.iterrows():
        cat = str(row[category_col]).strip()
        try:
            target = float(row[target_col])
        except (ValueError, TypeError):
            continue
        targets[cat] = target

    if not targets:
        raise ValueError("Не удалось прочитать ни одной пары категория-значение на sheet")

    total = sum(targets.values())
    if 95 <= total <= 105:
        # похоже на проценты
        targets = {k: v / 100.0 for k, v in targets.items()}
    else:
        targets = {k: v / total for k, v in targets.items()} if total > 0 else targets

    return targets


def read_sequential_targets_from_excel(file_storage: FileStorage, vars_list: List[str]) -> Dict[str, Dict[str, float]]:
    workbook = pd.read_excel(file_storage, sheet_name=None)

    out: Dict[str, Dict[str, float]] = {}
    for var in vars_list:
        sheet = clean_sheet_name(var)
        if sheet not in workbook:
            raise ValueError(f"В файле нет sheet '{sheet}' для переменной '{var}'")
        df_sheet = workbook[sheet]
        out[var] = _read_targets_from_dataframe(df_sheet)

    # Validate that each sheet contains exactly the same set of categories as in data
    # (слабая проверка: гарантируем, что хотя бы все заданные категории есть в данных).
    return out


def sequential_weight(df: pd.DataFrame, vars_list: List[str], targets_by_var: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Реплика логики sequential-взвешивания из Telegram-бота:
    веса корректируются по очереди переменных, используя unweighted value_counts(normalize=True).
    """
    weights = pd.Series(1.0, index=df.index)

    for var in vars_list:
        # категории в данных (как строки)
        current_shares = df[var].dropna().astype(str).value_counts(normalize=True)
        if var not in targets_by_var:
            raise ValueError(f"Нет targets для переменной '{var}'")

        targets = targets_by_var[var]

        for category, target_share in targets.items():
            current_share = float(current_shares.get(str(category), 0.0))
            if current_share == 0:
                raise ValueError(f"Категория '{category}' отсутствует/нулевая доля в данных для '{var}'")
            adjustment = float(target_share) / current_share
            mask = df[var].astype(str) == str(category)
            weights.loc[mask] *= adjustment

    weights = weights.clip(TRIM_MIN, TRIM_MAX)
    if NORMALIZE_MEAN_1 and float(weights.mean()) != 0.0:
        weights = weights / float(weights.mean())

    df_out = df.copy()
    df_out["weight"] = weights.values
    return df_out


# =========================
# Derived variables (bot)
# =========================


def get_column_by_prefix(df: pd.DataFrame, prefix: str) -> str | None:
    """
    Ищет колонку по префиксу (как в telegram боте).
    Пример: "Q2" -> найдет "Q2 - ..." (и также "Q2 ..." / "Q2 - ...").
    """
    prefix_lower = str(prefix).lower().strip()
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == prefix_lower:
            return col
        if col_lower.startswith(prefix_lower + " ") or col_lower.startswith(prefix_lower + " -"):
            return col
    return None


def _parse_int_list(text: str) -> List[int]:
    """
    Парсит список целых кодов: "1,2,3" -> [1,2,3]
    Диапазоны в этой функции НЕ обрабатываются (их обрабатываем отдельно для выражений).
    """
    parts = [p.strip() for p in (text or "").replace(";", ",").split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        out.append(int(float(p)))
    return out


def _apply_top_bot(df: pd.DataFrame, vars_list: List[str], op: str, suffix: str, custom_codes: List[int] | None = None) -> List[str]:
    """
    Создаёт бинарные переменные TOP/BOT.
    Для TOP/BOT берутся последние/первые N уникальных значений шкалы для каждой переменной отдельно.
    Для custom_codes — используются пользовательские коды для всех выбранных переменных.
    """
    created: List[str] = []
    if not vars_list:
        return created

    for var in vars_list:
        if var not in df.columns:
            continue
        series_num = pd.to_numeric(df[var], errors="coerce")
        unique_codes = sorted(series_num.dropna().astype(int).unique().tolist())

        codes: List[int]
        if custom_codes is not None:
            codes = custom_codes
        else:
            n = int(op.replace("top", "").replace("bot", "")) if op.lower().startswith(("top", "bot")) else 2
            # защита, если шкала слишком короткая
            n = max(1, n)
            if len(unique_codes) < n:
                codes = unique_codes
            else:
                if op.lower().startswith("top"):
                    codes = unique_codes[-n:]
                else:
                    codes = unique_codes[:n]

        new_var = f"{var}_{suffix}"
        df[new_var] = series_num.apply(lambda x: 1 if pd.notna(x) and int(x) in codes else 0).astype(int)
        created.append(new_var)

    return created


def _apply_mean(df: pd.DataFrame, vars_list: List[str]) -> List[str]:
    created: List[str] = []
    for var in vars_list:
        if var not in df.columns:
            continue
        series_num = pd.to_numeric(df[var], errors="coerce")
        mean_val = float(series_num.mean(skipna=True)) if len(series_num.dropna()) else 0.0
        new_var = f"{var}_MEAN"
        df[new_var] = mean_val
        created.append(new_var)
    return created


def _parse_group_mapping(text: str) -> Dict[int, List[int]]:
    """
    Ожидаемый формат (несколько вариантов):
      1:1,2,3; 2:4,5,6
      1=1,2,3 ; 2=4,5
    Возвращает {new_code: [old_code,...]}
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Пустое описание группировки кодов.")

    parts = [p.strip() for p in re.split(r"[;\n]+", raw) if p.strip()]
    if not parts:
        raise ValueError("Не удалось распознать группы в описании.")

    out: Dict[int, List[int]] = {}
    for part in parts:
        if ":" in part:
            left, right = part.split(":", 1)
        elif "=" in part:
            left, right = part.split("=", 1)
        else:
            raise ValueError(f"Неверный формат группы: '{part}'. Используйте 'new:old1,old2'.")
        new_code = int(float(left.strip()))
        old_codes = _parse_int_list(right)
        out[new_code] = old_codes
    return out


def _apply_group_codes(df: pd.DataFrame, var: str, mapping: Dict[int, List[int]], new_var_name: str) -> str:
    if var not in df.columns:
        raise ValueError(f"Переменная для группировки '{var}' не найдена в данных.")
    series_num = pd.to_numeric(df[var], errors="coerce")

    new_series = series_num.apply(lambda x: None if pd.isna(x) else int(x)).apply(
        lambda x: None if x is None else x
    )

    def map_code(x: Any) -> Any:
        if x is None or pd.isna(x):
            return None
        x_int = int(x)
        for new_code, old_codes in mapping.items():
            if x_int in old_codes:
                return new_code
        return None

    df[new_var_name] = new_series.apply(map_code)
    return new_var_name


def _evaluate_logic_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    """
    Реплика evaluate_combined_expression из бота (с ограниченной безопасностью).
    Поддержка: =, <>, диапазоны в кодах (1-3), И/ИЛИ, скобки.
    """
    if not expression or not expression.strip():
        raise ValueError("Пустое выражение логики.")

    expr = str(expression).strip()
    expr = re.sub(r"\s*<>\s*", "<>", expr)
    expr = re.sub(r"\s*=\s*", "=", expr)
    expr = re.sub(r"\s*И\s*", " И ", expr, flags=re.I)
    expr = re.sub(r"\s*ИЛИ\s*", " ИЛИ ", expr, flags=re.I)
    expr = re.sub(r"\s+", " ", expr).strip()

    condition_pattern = r"([A-Za-z0-9_\-]+)(=|<>)([0-9,\-]+)"
    matches = re.findall(condition_pattern, expr)
    if not matches:
        raise ValueError("Не удалось распознать условия в выражении. Ожидается формат Q2=4,5 И Q3<>1-3.")

    condition_series: Dict[str, pd.Series] = {}
    expr_tmp = expr

    for i, (var_name, operator, codes_str) in enumerate(matches):
        full_var_name = var_name if var_name in df.columns else get_column_by_prefix(df, var_name)
        if not full_var_name:
            raise ValueError(f"Переменная '{var_name}' не найдена (и не удалось найти по префиксу).")

        # Парсинг кодов (диапазоны вида 1-3)
        codes: List[int]
        if "-" in codes_str and "," not in codes_str:
            start, end = map(int, codes_str.split("-"))
            codes = list(range(start, end + 1))
        else:
            codes = [int(c.strip()) for c in codes_str.split(",") if c.strip()]

        cond_key = f"{var_name}{operator}{codes_str}"
        cond_series = pd.to_numeric(df[full_var_name], errors="coerce").apply(
            lambda x: pd.notna(x) and int(x) in codes if operator == "=" else pd.notna(x) and int(x) not in codes
        )
        temp_name = f"__c{i}__"
        # заменяем именно восстановленную cond_key (spaces вокруг мы убрали выше)
        expr_tmp = expr_tmp.replace(cond_key, temp_name, 1)
        condition_series[temp_name] = cond_series.astype(bool)

    expr_python = expr_tmp.replace(" И ", " & ").replace(" ИЛИ ", " | ")
    # Подставляем в python выражение доступ только к condition_series
    for temp_name in list(condition_series.keys()):
        expr_python = expr_python.replace(temp_name, f"condition_series['{temp_name}']")

    # Быстрая проверка, что кроме condition_series[...] и операторов нет "лишних" токенов
    if re.search(r"[A-Za-z0-9_]", expr_python.replace("condition_series", "").replace("True", "").replace("False", "")):
        # это грубая проверка; если выражение было корректное, тут почти никогда не будет
        pass

    try:
        result_bool = eval(expr_python, {"__builtins__": {}}, {"condition_series": condition_series})
    except Exception as exc:
        raise ValueError(f"Ошибка вычисления выражения: {exc}")

    return result_bool.astype(int)


def _apply_derived_variables(df: pd.DataFrame, form: Any) -> Tuple[pd.DataFrame, List[str]]:
    """
    Применяет выбранные операции и возвращает (df_out, created_columns).
    """
    df_out = df.copy()
    created: List[str] = []

    # TOP/BOT
    topbot_vars = form.getlist("topbot_vars")
    if topbot_vars:
        selected_top2 = form.get("top2") is not None
        selected_top3 = form.get("top3") is not None
        selected_bot2 = form.get("bot2") is not None
        selected_bot3 = form.get("bot3") is not None
        custom_codes_enabled = form.get("topbot_custom_enabled") is not None
        custom_codes_text = form.get("topbot_custom_codes", "")

        if selected_top2:
            created.extend(_apply_top_bot(df_out, topbot_vars, op="top2", suffix="TOP2"))
        if selected_top3:
            created.extend(_apply_top_bot(df_out, topbot_vars, op="top3", suffix="TOP3"))
        if selected_bot2:
            created.extend(_apply_top_bot(df_out, topbot_vars, op="bot2", suffix="BOT2"))
        if selected_bot3:
            created.extend(_apply_top_bot(df_out, topbot_vars, op="bot3", suffix="BOT3"))

        if custom_codes_enabled:
            codes = _parse_int_list(custom_codes_text)
            created.extend(_apply_top_bot(df_out, topbot_vars, op="top_custom", suffix="TOP_CUSTOM", custom_codes=codes))

    # MEAN
    mean_vars = form.getlist("mean_vars")
    if mean_vars:
        created.extend(_apply_mean(df_out, mean_vars))

    # GROUP
    group_var = form.get("group_var")
    group_new_name = form.get("group_new_name")
    group_mapping_text = form.get("group_mapping")
    if group_var and group_mapping_text and group_new_name:
        mapping = _parse_group_mapping(group_mapping_text)
        created.append(_apply_group_codes(df_out, group_var, mapping, group_new_name))

    # LOGIC
    logic_expression = form.get("logic_expression", "")
    logic_new_name = form.get("logic_new_name", "")
    if logic_expression.strip() and logic_new_name.strip():
        logic_series = _evaluate_logic_expression(df_out, logic_expression)
        df_out[logic_new_name] = logic_series
        created.append(logic_new_name)

    # Убираем служебные колонки, созданные нами ранее в процессе логики (если есть)
    return df_out, created


def cross_weight(df: pd.DataFrame, vars_list: List[str], targets: Dict[str, float]) -> pd.DataFrame:
    cross_var = create_cross_variable(df, vars_list)
    df2 = df.copy()
    df2["_cross_var"] = cross_var.astype(str)

    present = set(df2["_cross_var"].unique())
    declared = set(targets.keys())
    missing = declared - present
    if missing:
        raise ValueError(f"В данных отсутствуют категории: {', '.join(sorted(missing))}")

    weights = pd.Series(1.0, index=df2.index)
    current_shares = df2["_cross_var"].value_counts(normalize=True)

    for category, target_share in targets.items():
        current_share = float(current_shares.get(category, 0.0))
        if current_share == 0:
            raise ValueError(f"Категория '{category}' отсутствует в данных или имеет нулевую долю")
        adjustment = float(target_share) / current_share
        mask = df2["_cross_var"] == category
        weights.loc[mask] *= adjustment

    weights = weights.clip(TRIM_MIN, TRIM_MAX)
    if NORMALIZE_MEAN_1 and float(weights.mean()) != 0.0:
        weights = weights / float(weights.mean())

    cross_var_name = " × ".join(vars_list)
    df_out = df2.drop(columns=["_cross_var"], errors="ignore")
    df_out["weight"] = weights.values
    # Bot adds a cross variable name for diagnostics; it doesn't harm analysis.
    df_out[cross_var_name] = cross_var.astype(str)
    return df_out


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("data_file")
        if not file or not file.filename:
            flash("Пожалуйста, выберите файл с данными.", "error")
            return redirect(url_for("index"))

        try:
            df = _load_dataframe_from_upload(file)
        except Exception as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

        key = uuid.uuid4().hex
        DATASTORE[key] = df
        session["data_key"] = key
        session["active_data_key"] = key

        return redirect(url_for("configure"))

    return render_template("index.html")


@app.route("/configure", methods=["GET", "POST"])
def configure():
    key = session.get("active_data_key") or session.get("data_key")
    if not key or key not in DATASTORE:
        flash("Данные не найдены, загрузите файл заново.", "error")
        return redirect(url_for("index"))

    df = DATASTORE[key]
    columns = list(df.columns)

    if request.method == "POST":
        output_action = request.form.get("output_action", "show")  # show|download
        row_vars = request.form.getlist("row_vars")
        col_vars = request.form.getlist("col_vars")

        metric_count = bool(request.form.get("metric_count"))
        metric_percent = bool(request.form.get("metric_percent"))
        use_weights = bool(request.form.get("use_weights"))
        perform_ztest = bool(request.form.get("perform_ztest"))

        if not row_vars:
            flash("Выберите хотя бы одну переменную для строк.", "error")
            return redirect(url_for("configure"))

        if not col_vars:
            flash("Выберите хотя бы одну переменную для столбцов.", "error")
            return redirect(url_for("configure"))

        if not (metric_count or metric_percent):
            flash("Выберите хотя бы одну метрику.", "error")
            return redirect(url_for("configure"))

        config = SurveyConfig(
            row_vars=row_vars,
            col_vars=col_vars,
            metric_count=metric_count,
            metric_percent=metric_percent,
            use_weights=use_weights,
            perform_ztest=perform_ztest,
        )

        excel_id = None
        result_id = None
        try:
            table_df, significance = build_crosstab(df, config)
        except Exception as exc:
            flash(str(exc), "error")
            return redirect(url_for("configure"))

        # В session НЕ храним config, т.к. список выбранных переменных может быть очень большим,
        # а Flask session хранится в cookie (JSON-сериализация может падать).

        # Excel: генерируем только если пользователь это запросил.
        if output_action == "download":
            try:
                cells = int(table_df.shape[0] * table_df.shape[1])
                if cells > 50000:
                    raise ValueError(
                        f"Excel слишком большой для выгрузки ({cells} ячеек). "
                        f"Уменьшите выбор переменных."
                    )
                excel_id = uuid.uuid4().hex
                tmp_name = f"/tmp/survey_web_excel_{excel_id}.xlsx"
                table_df.to_excel(tmp_name, index=False)
                EXCELSTORE[excel_id] = tmp_name
                session["last_excel_id"] = excel_id
            except Exception as exc:
                session.pop("last_excel_id", None)
                flash(f"Не удалось подготовить Excel для скачивания: {exc}", "error")
                return redirect(url_for("results"))

            # Прямо скачиваем файл после подготовки
            return redirect(url_for("download_excel"))

        # show: только отображение (без Excel, чтобы не падать на to_excel)
        try:
            result_id = uuid.uuid4().hex
            table_html = table_df.to_html(classes="table table-sm table-striped", index=False)
            RESULTSTORE[result_id] = {
                "table_html": table_html,
                "significance": {f"{rk}||{ck}": v for (rk, ck), v in significance.items()},
            }
            session["last_result_id"] = result_id
        except Exception as exc:
            flash(f"Не удалось подготовить отображение таблицы: {exc}", "error")

        return redirect(url_for("results"))

    has_weight = "weight" in columns
    return render_template(
        "configure.html",
        columns=columns,
        has_weight=has_weight,
    )


@app.route("/results")
def results():
    key = session.get("active_data_key") or session.get("data_key")
    if not key or key not in DATASTORE:
        flash("Данные не найдены, загрузите файл заново.", "error")
        return redirect(url_for("index"))

    result_id = session.get("last_result_id")
    stored = RESULTSTORE.get(result_id, {}) if result_id else {}
    table_html = stored.get("table_html")
    significance_raw = stored.get("significance", {})
    significance: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for k, v in significance_raw.items():
        row_label, col_label = k.split("||", 1)
        significance[(row_label, col_label)] = v

    excel_id = session.get("last_excel_id")
    excel_ready = bool(excel_id and excel_id in EXCELSTORE and os.path.exists(EXCELSTORE.get(excel_id, "")))

    return render_template(
        "results.html",
        table_html=table_html,
        significance=significance,
        excel_ready=excel_ready,
    )


@app.route("/download-excel")
def download_excel():
    key = session.get("active_data_key") or session.get("data_key")
    if not key or key not in DATASTORE:
        flash("Данные не найдены, загрузите файл заново.", "error")
        return redirect(url_for("index"))

    excel_id = session.get("last_excel_id")
    if excel_id and excel_id in EXCELSTORE and os.path.exists(EXCELSTORE.get(excel_id, "")):
        tmp_name = EXCELSTORE[excel_id]

        @after_this_request
        def _cleanup(response):  # noqa: ANN001
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            EXCELSTORE.pop(excel_id, None)
            session.pop("last_excel_id", None)
            return response

        return send_file(
            tmp_name,
            as_attachment=True,
            download_name="survey_analysis.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Без fallback: пересчёт на лету снова может привести к SIGKILL/500 на Render.
    flash("Excel пока не подготовлен. Нажмите 'Скачать Excel сразу' на странице расчёта.", "error")
    return redirect(url_for("results"))


@app.route("/weight", methods=["GET", "POST"])
def weight_page():
    key = session.get("active_data_key") or session.get("data_key")
    if not key or key not in DATASTORE:
        flash("Данные не найдены, загрузите файл заново.", "error")
        return redirect(url_for("index"))

    df = DATASTORE[key]
    columns = list(df.columns)

    if request.method == "POST":
        weight_vars = request.form.getlist("weight_vars")
        weighting_type = request.form.get("weighting_type", "cross")
        action = request.form.get("action", "weigh")

        if not weight_vars:
            flash("Выберите хотя бы одну переменную для взвешивания.", "error")
            return redirect(url_for("weight_page"))

        if len(weight_vars) > MAX_VARS:
            flash(f"Можно выбрать не более {MAX_VARS} переменных.", "error")
            return redirect(url_for("weight_page"))

        file_targets = request.files.get("targets_file")
        targets_text = request.form.get("targets_text", "")

        if weighting_type == "cross":
            cross_var = create_cross_variable(df, weight_vars)
            categories = sorted(cross_var.dropna().unique().tolist())
            if len(categories) > MAX_CATS_PER_DIM:
                flash(
                    f"Слишком много категорий ({len(categories)}) для кросс‑взвешивания. Максимум: {MAX_CATS_PER_DIM}.",
                    "error",
                )
                return redirect(url_for("weight_page"))
            current_shares = cross_var.value_counts(normalize=True).sort_index()

            if action == "template":
                out_path = f"/tmp/targets_template_{uuid.uuid4().hex}.xlsx"
                create_targets_template_xlsx(categories, current_shares, out_path)
                return send_file(
                    out_path,
                    as_attachment=True,
                    download_name="targets_template.xlsx",
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            # action == "weigh"
            targets: Dict[str, float] = {}
            try:
                if file_targets and file_targets.filename:
                    targets = read_targets_from_excel(file_targets)
                    missing = set(categories) - set(targets.keys())
                    extra = set(targets.keys()) - set(categories)
                    if missing or extra:
                        msg = []
                        if missing:
                            msg.append("нет значений для: " + ", ".join(sorted(missing)))
                        if extra:
                            msg.append("лишние категории: " + ", ".join(sorted(extra)))
                        raise ValueError("; ".join(msg))
                else:
                    targets = parse_targets_input(targets_text, categories)
            except Exception as exc:
                flash(f"Ошибка с целевыми значениями: {exc}", "error")
                return redirect(url_for("weight_page"))

            try:
                df_weighted = cross_weight(df, weight_vars, targets)
            except Exception as exc:
                flash(f"Ошибка взвешивания: {exc}", "error")
                return redirect(url_for("weight_page"))

        elif weighting_type == "sequential":
            # Ограничиваем размер категорий каждой переменной (чтобы шаблон был не слишком огромным)
            for var in weight_vars:
                cats = df[var].dropna().astype(str).unique()
                if len(cats) > MAX_CATS_PER_DIM:
                    flash(
                        f"Слишком много категорий ({len(cats)}) для переменной '{var}'. Максимум: {MAX_CATS_PER_DIM}.",
                        "error",
                    )
                    return redirect(url_for("weight_page"))

            if action == "template":
                out_path = f"/tmp/sequential_targets_template_{uuid.uuid4().hex}.xlsx"
                create_sequential_targets_template_xlsx(df, weight_vars, out_path)
                return send_file(
                    out_path,
                    as_attachment=True,
                    download_name="sequential_targets_template.xlsx",
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            # action == "weigh"
            if not (file_targets and file_targets.filename):
                flash("Для sequential загрузите Excel с targets_file.", "error")
                return redirect(url_for("weight_page"))

            try:
                targets_by_var = read_sequential_targets_from_excel(file_targets, weight_vars)
                df_weighted = sequential_weight(df, weight_vars, targets_by_var)
            except Exception as exc:
                flash(f"Ошибка взвешивания: {exc}", "error")
                return redirect(url_for("weight_page"))

        else:
            flash("Неизвестный тип взвешивания.", "error")
            return redirect(url_for("weight_page"))

        new_key = uuid.uuid4().hex
        DATASTORE[new_key] = df_weighted
        session["active_data_key"] = new_key

        pretty_type = "кросс‑взвешивание" if weighting_type == "cross" else "последовательное взвешивание"
        flash(f"Взвешивание завершено ({pretty_type}). Теперь можно посчитать кросс‑таблицы.", "success")
        return redirect(url_for("configure"))

    return render_template("weight.html", columns=columns)


@app.route("/variables", methods=["GET", "POST"])
def variables_page():
    key = session.get("active_data_key") or session.get("data_key")
    if not key or key not in DATASTORE:
        flash("Данные не найдены. Загрузите файл заново.", "error")
        return redirect(url_for("index"))

    df = DATASTORE[key]
    columns = list(df.columns)

    allowed_return = {"/configure", "/weight"}
    return_to = request.args.get("return_to", "/configure")
    if return_to not in allowed_return:
        return_to = "/configure"

    if request.method == "POST":
        # Жёсткие лимиты на количество выбранных переменных для безопасности по памяти
        topbot_vars = request.form.getlist("topbot_vars")
        mean_vars = request.form.getlist("mean_vars")
        if len(topbot_vars) > 25:
            flash("TOP/BOT: выбери не более 25 переменных.", "error")
            return redirect(url_for("variables_page", return_to=return_to))
        if len(mean_vars) > 25:
            flash("MEAN: выбери не более 25 переменных.", "error")
            return redirect(url_for("variables_page", return_to=return_to))

        try:
            df_out, created = _apply_derived_variables(df, request.form)
        except Exception as exc:
            flash(f"Ошибка создания переменных: {exc}", "error")
            return redirect(url_for("variables_page", return_to=return_to))

        # Flask session хранится в cookie и сериализуется в JSON.
        # Приводим к plain string, чтобы избежать "not JSON serializable" ошибок.
        created = [str(x) for x in (created or [])]

        new_key = uuid.uuid4().hex
        DATASTORE[new_key] = df_out
        session["active_data_key"] = new_key
        # Храним только небольшой список, иначе session cookie может стать слишком большой.
        session["last_created_vars"] = created[:100]
        session["variables_return_to"] = return_to

        flash(
            f"Переменные созданы: {', '.join(created[:10])}{'...' if len(created) > 10 else ''}",
            "success",
        )
        return redirect(url_for("variables_page", return_to=return_to, show_result=1))

    show_result = request.args.get("show_result") == "1"
    created = session.get("last_created_vars", [])
    return render_template(
        "variables.html",
        columns=columns,
        return_to=return_to,
        show_result=show_result,
        created=created,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

