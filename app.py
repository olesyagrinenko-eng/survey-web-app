from __future__ import annotations

import os
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
    use_nested_columns: bool = False
    show_sig_marks: bool = True
    use_nested_columns: bool = False


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

    # Подготовка столбцов анализа.
    # По умолчанию - один уровень: каждая выбранная переменная раскрывается отдельно.
    # Если включен use_nested_columns и выбрано 2+ переменных - строим комбинации.
    analysis_cols: List[Tuple[pd.Series | None, str]] = []  # (mask, label)

    # Total сначала
    analysis_cols.append((None, "Total"))

    if config.use_nested_columns and len(config.col_vars) > 1:
        combos = df[config.col_vars].dropna().drop_duplicates()
        if not combos.empty:
            combos = combos.assign(
                __sort_key__=combos.apply(lambda r: " | ".join([str(r[c]) for c in config.col_vars]), axis=1)
            ).sort_values("__sort_key__")
            for _, combo in combos.iterrows():
                mask = pd.Series(True, index=df.index)
                parts = []
                for col_var in config.col_vars:
                    val = combo[col_var]
                    mask = mask & (df[col_var] == val)
                    parts.append(f"{col_var}: {val}")
                analysis_cols.append((mask, " | ".join(parts)))
    else:
        for col_var in config.col_vars:
            uniques = _safe_sort(df[col_var].dropna().unique())
            for val in uniques:
                label = f"{col_var}: {val}"
                analysis_cols.append((df[col_var] == val, label))

    # заголовки
    headers = ["Переменная"]
    headers.extend([lbl for _, lbl in analysis_cols])

    rows: List[List[Any]] = []
    significance: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # первая строка — невзвешенные размеры выборки
    sample_row = ["Выборка (N)"]
    for mask, _lbl in analysis_cols:
        n = len(df) if mask is None else int(mask.sum())
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

            for idx, (mask, _lbl) in enumerate(analysis_cols):
                if mask is None:
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
                    col_mask = mask
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
            if config.perform_ztest and config.metric_percent and total_base > 0:
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
                    col_label = analysis_cols[idx][1]
                    significance[(label, col_label)] = {
                        "z": z,
                        "p": p,
                        "direction": direction,
                    }

            rows.append(row)

    table_df = pd.DataFrame(rows, columns=headers)
    return table_df, significance


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

        return redirect(url_for("configure"))

    return render_template("index.html")


@app.route("/configure", methods=["GET", "POST"])
def configure():
    key = session.get("data_key")
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
        use_nested_columns = bool(request.form.get("use_nested_columns"))

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
            use_nested_columns=use_nested_columns,
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
    key = session.get("data_key")
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

