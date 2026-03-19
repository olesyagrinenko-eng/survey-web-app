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

