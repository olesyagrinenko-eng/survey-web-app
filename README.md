## Веб‑интерфейс для анализа опросных данных

Это упрощённая веб‑версия Telegram‑бота `survey_analysis_bot_FINAL.py`.

### Что умеет

- **Загрузка файлов**: `CSV`, `XLS`, `XLSX`, `SAV (SPSS)`.
- **Выбор переменных**:
  - для **строк** (вопросы/значения);
  - для **столбцов** (группы/сегменты, в т.ч. Total).
- **Метрики**:
  - количество (`N`);
  - процент по столбцу.
- **Веса**: опциональное использование столбца `weight` (как в боте).
- **Z‑тест против Total**:
  - классический пропорционный z‑тест;
  - для каждой группы сравнивает долю с `Total`, считает `z` и `p`;
  - выводит список ячеек, где `p < 0.05` (выше или ниже Total).
- **Экспорт в Excel**: выгрузка рассчитанной таблицы в `survey_analysis.xlsx`.

### Локальный запуск

```bash
cd web-survey-app
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export FLASK_SECRET_KEY="your-secret"  # опционально
python app.py
```

Откройте в браузере: `http://localhost:5000`.

### Деплой на Render

1. Выложите папку `web-survey-app` в GitHub.
2. В Render создайте **Web Service**:
   - Environment: `Python`;
   - Root directory: `web-survey-app`;
   - Build command: `pip install -r requirements.txt`;
   - Start command: `gunicorn app:app`.
3. (Опционально) задайте переменную `FLASK_SECRET_KEY`.

### Логи: `Worker was sent code 139`

Код **139** в gunicorn означает **SIGSEGV** — процесс упал из‑за ошибки в **нативном** коде (не из‑за исключения Python). Чаще всего это связано с чтением **SPSS** через **pyreadstat**/ReadStat на конкретном файле или нехваткой памяти на инстансе.

Что сделать:

- Задеплойте актуальные зависимости (`pyreadstat` в `requirements.txt` обновляйте вместе с релизами).
- Для больших `.sav` возьмите инстанс Render с **большим объёмом RAM**.
- В **Environment** можно задать `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1` (как в `render.yaml`) — меньше пиков памяти при расчётах.
- Если падает только на одном файле — проверьте файл в SPSS или откройте локально тем же `pyreadstat`; при необходимости экспортируйте в CSV/XLSX и загрузите его.

