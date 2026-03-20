from __future__ import annotations

import os
import re
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

try:
    import pyreadstat  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    pyreadstat = None


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")


# --- Simple in‑memory "storage" for uploaded dataframes (per session) ---
DATASTORE: Dict[str, pd.DataFrame] = {}


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
        df, _meta = pyreadstat.read_sav(file_storage)
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

    # Подготовка столбцов анализа (Total + все выбранные колоночные переменные)
    analysis_cols: List[Tuple[str | None, Any | None, str]] = []  # (col_var, col_val, label)

    # Total сначала
    analysis_cols.append((None, None, "Total"))

    for col_var in config.col_vars:
        uniques = _safe_sort(df[col_var].dropna().unique())
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
        # для очень "широких" переменных не ограничиваем жёстко, но это может быть тяжело
        for rv in unique_vals:
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

        table_df, significance = build_crosstab(df, config)
        session["last_config"] = config.__dict__

        # сохраняем промежуточно
        session["last_table_html"] = table_df.to_html(classes="table table-sm table-striped", index=False)
        session["last_significance"] = {
            f"{rk}||{ck}": v for (rk, ck), v in significance.items()
        }

        return redirect(url_for("results"))

    has_weight = "weight" in columns
    return render_template(
        "configure.html",
        columns=columns,
        has_weight=has_weight,
    )


@app.route("/results")
def results():
    key = session.get("data_key")
    if not key or key not in DATASTORE:
        flash("Данные не найдены, загрузите файл заново.", "error")
        return redirect(url_for("index"))

    table_html = session.get("last_table_html")
    significance_raw = session.get("last_significance", {})
    significance: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for k, v in significance_raw.items():
        row_label, col_label = k.split("||", 1)
        significance[(row_label, col_label)] = v

    return render_template(
        "results.html",
        table_html=table_html,
        significance=significance,
    )


@app.route("/download-excel")
def download_excel():
    key = session.get("active_data_key") or session.get("data_key")
    if not key or key not in DATASTORE:
        flash("Данные не найдены, загрузите файл заново.", "error")
        return redirect(url_for("index"))

    df = DATASTORE[key]
    cfg_dict = session.get("last_config")
    if not cfg_dict:
        flash("Сначала рассчитайте таблицу.", "error")
        return redirect(url_for("configure"))

    config = SurveyConfig(**cfg_dict)
    table_df, _sig = build_crosstab(df, config)

    tmp_name = f"/tmp/survey_web_{uuid.uuid4().hex}.xlsx"
    table_df.to_excel(tmp_name, index=False)

    return send_file(
        tmp_name,
        as_attachment=True,
        download_name="survey_analysis.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/weight", methods=["GET", "POST"])
def weight_page():
    key = session.get("data_key")
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

