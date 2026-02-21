# bg_remove.py — подробный и простой гайд

Этот файл объясняет, как пользоваться `bg_remove.py`, чтобы получать максимально хороший результат и не упираться в ошибки по памяти на CPU.

## Что делает скрипт

`bg_remove.py`:
1. Берет вход: один файл или целую папку (рекурсивно).
2. Уменьшает изображение перед удалением фона (если включено в константах).
3. Удаляет фон через `rembg`.
4. Применяет постобработку краев (по настройкам).
5. Сохраняет результат в PNG.

## Важный принцип качества

На качество сильнее всего влияют:
1. Размер изображения перед обработкой.
2. Профиль/модель.
3. Alpha matting.
4. Постобработка краев.

Если скрипт «съедает» лишнее, обычно надо сначала править размер и профиль, а уже потом тонко крутить matting.

## Текущие дефолты в скрипте

Верхние константы в `bg_remove.py` сейчас такие:
1. `DEFAULT_ALPHA_MATTING = False`
2. `DEFAULT_FALLBACK_MODE = "off"`
3. `DEFAULT_PROFILE = "auto"`
4. `DEFAULT_PRE_RESIZE_MAX_SIDE = 300`
5. `DEFAULT_PRE_RESIZE_ALIGN = 16`
6. `DEFAULT_MAX_INFERENCE_PIXELS = 16000000`
7. `DEFAULT_ALPHA_MATTING_MAX_PIXELS = 12000000`

Это означает:
1. По умолчанию режим ориентирован на стабильность CPU.
2. Картинки сначала уменьшаются до длинной стороны около `300 px`.
3. Fallback на вторую модель по умолчанию выключен.

## Быстрый старт

Запуск с дефолтами:

```powershell
& C:/Python313/python.exe d:/LetsProgram/python-vscode/TEST_PROJECTS/splitter/bg_remove.py
```

Если нужно указать свои папки:

```powershell
& C:/Python313/python.exe d:/LetsProgram/python-vscode/TEST_PROJECTS/splitter/bg_remove.py "d:/in_folder" "d:/out_folder"
```

## Установка зависимостей (CPU)

Рекомендуемый набор для слабой видеокарты:

```powershell
pip uninstall -y onnxruntime-gpu
pip install --upgrade rembg onnxruntime pillow
```

## Как скрипт выбирает и обрабатывает

Пайплайн на один файл:
1. Чтение изображения.
2. Pre-resize по `DEFAULT_PRE_RESIZE_MAX_SIDE`.
3. Доп. уменьшение при слишком большом размере (`--max-inference-pixels`).
4. Выбор профиля (`--profile`), модельный прогон.
5. При OOM в matting идет автоповтор с `alpha_matting=False`.
6. Постобработка (`--edge-refine-radius`, `--decontaminate`).
7. Сохранение PNG.

## Главные параметры и когда их крутить

`DEFAULT_PRE_RESIZE_MAX_SIDE`:
1. `300` часто дает аккуратнее маску на людях.
2. `512` или `768` лучше сохраняет детали предметов.
3. `None` или `""` отключает предуменьшение.

`DEFAULT_PRE_RESIZE_ALIGN`:
1. `16` выравнивает размеры до «красивых» значений.
2. Пример: `608x416 -> 304x208`.
3. Поставь `1`, если не нужно выравнивание.

`--profile`:
1. `auto` универсально.
2. `general` часто бережнее к объекту.
3. `portrait` для людей.
4. `product` для товарных кадров.

`--fallback-mode`:
1. `off` быстрее, стабильнее, но без второй попытки.
2. `smart` часто дает лучше качество.
3. `always` медленнее всего, но максимально «перестраховывает».

`DEFAULT_ALPHA_MATTING` и `--no-alpha-matting`:
1. Matting улучшает сложные края, но ест память.
2. Сейчас по умолчанию matting выключен (`DEFAULT_ALPHA_MATTING=False`).
3. Если хочешь включить matting по умолчанию, поставь `DEFAULT_ALPHA_MATTING=True`.

`--af --ab --ae` (когда matting включен):
1. Более бережный старт: `--af 220 --ab 5 --ae 0`.
2. Большой `ae` может подрезать край объекта.

`--edge-refine-radius`:
1. `0` выключить сглаживание.
2. `0.3-1.0` обычно достаточно.
3. Слишком большое значение может «размылить» мелкие детали.

`--decontaminate`:
1. `off` если искажается цвет по краю.
2. `light` как мягкий безопасный вариант.
3. `medium/strong` только если явно видны цветные ореолы.

## Практичные пресеты

Стабильный CPU-режим:

```powershell
& C:/Python313/python.exe d:/LetsProgram/python-vscode/TEST_PROJECTS/splitter/bg_remove.py --no-alpha-matting --fallback-mode off
```

Более качественный универсальный:

```powershell
& C:/Python313/python.exe d:/LetsProgram/python-vscode/TEST_PROJECTS/splitter/bg_remove.py --profile general --fallback-mode smart --edge-refine-radius 0.3 --decontaminate off
```

Когда «съедает» лишнее:

```powershell
& C:/Python313/python.exe d:/LetsProgram/python-vscode/TEST_PROJECTS/splitter/bg_remove.py --profile general --fallback-mode smart --edge-refine-radius 0 --decontaminate off
```

И дополнительно вверху скрипта подними:
1. `DEFAULT_PRE_RESIZE_MAX_SIDE` с `300` до `512` или `768`.

Когда человек с мелкими деталями наоборот лучше на маленьком размере:
1. Оставь `DEFAULT_PRE_RESIZE_MAX_SIDE = 300`.
2. Оставь `DEFAULT_PRE_RESIZE_ALIGN = 16`.
3. Используй профиль `auto` или `portrait`.

## Частые сообщения и что с ними делать

`PERFORMANCE WARNING ... Cholesky ...`:
1. Обычно не критично.
2. Если результат нормальный, можно игнорировать.

`Unable to allocate ...`:
1. Это нехватка RAM.
2. Снизь размер входа (`DEFAULT_PRE_RESIZE_MAX_SIDE`), либо отключи matting.
3. В скрипте уже есть авто-повтор без matting при OOM.

`cublasLt64_12.dll is missing`:
1. Это попытка CUDA при отсутствии нужных библиотек.
2. Для CPU просто используй `onnxruntime` без `onnxruntime-gpu`.

## Тихий лог ONNXRuntime

В PowerShell на текущую сессию:

```powershell
$env:ORT_LOG_SEVERITY_LEVEL="3"
```

Постоянно для пользователя:

```powershell
[System.Environment]::SetEnvironmentVariable("ORT_LOG_SEVERITY_LEVEL","3","User")
```

## Как правильно подбирать настройки (короткий алгоритм)

1. Запусти дефолтный режим.
2. Если «съедает» объект: увеличь `DEFAULT_PRE_RESIZE_MAX_SIDE` и поставь `--profile general`.
3. Если появляются ошибки памяти: `--no-alpha-matting`, `--fallback-mode off`.
4. Если края грубые: попробуй `--fallback-mode smart`, затем мягкий `--edge-refine-radius 0.3`.
5. Если цветной ореол: `--decontaminate light`, при искажении цвета обратно `off`.

## Какие файлы обрабатываются

Рекурсивно берутся:
1. `.png`
2. `.jpg`
3. `.jpeg`
4. `.webp`
5. `.bmp`
6. `.tiff`
7. `.tif`

Результат сохраняется в PNG, структура подпапок сохраняется.
