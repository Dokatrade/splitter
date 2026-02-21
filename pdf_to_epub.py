#!/usr/bin/env python3
"""
PDF to EPUB Converter (Improved Version)

Конвертирует PDF файлы в EPUB формат с улучшенным извлечением текста и изображений.

Использование:
    python pdf_to_epub.py                          # Обработает все PDF из d:\\results\\pdfepub
    python pdf_to_epub.py input.pdf                # Создаст input.epub
    python pdf_to_epub.py input.pdf -o output.epub # Указать имя выходного файла
    python pdf_to_epub.py input_folder -o output_folder  # Обработать папку с PDF файлами

Требуемые библиотеки:
    pip install PyMuPDF ebooklib Pillow
"""

import argparse
import os
import sys
import io
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Ошибка: Не установлена библиотека PyMuPDF")
    print("Установите её командой: pip install PyMuPDF")
    sys.exit(1)

try:
    from ebooklib import epub
except ImportError:
    print("Ошибка: Не установлена библиотека ebooklib")
    print("Установите её командой: pip install ebooklib")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Ошибка: Не установлена библиотека Pillow")
    print("Установите её командой: pip install Pillow")
    sys.exit(1)


# Путь по умолчанию для PDF файлов
DEFAULT_INPUT_PATH = r"d:\results\pdfepub"

# Количество страниц на главу
PAGES_PER_CHAPTER = 10


@dataclass
class TextBlock:
    """Блок текста с координатами и стилем."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float = 12.0
    is_bold: bool = False
    is_italic: bool = False


@dataclass
class ImageInfo:
    """Информация об изображении."""
    data: bytes
    extension: str
    width: int
    height: int
    y_position: float  # Позиция на странице для правильного порядка


@dataclass
class PageContent:
    """Содержимое страницы PDF."""
    page_num: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    images: List[ImageInfo] = field(default_factory=list)


def find_figure_regions(page: fitz.Page) -> List[Tuple[float, float, float, float, float]]:
    """
    Находит области с фигурами/графиками и формулами на странице.
    Ищет:
    - Области между подписями типа "Figure X:", "Chart X:" и "Note:", "Source:"
    - Формулы после предложений, заканчивающихся на двоеточие
    
    Returns:
        Список кортежей (x0, y0, x1, y1, y_position) с координатами областей
    """
    regions = []
    
    # Получаем все текстовые блоки
    blocks = page.get_text("dict").get("blocks", [])
    text_blocks = [b for b in blocks if b.get("type") == 0]
    
    # Паттерны для фигур
    figure_pattern = re.compile(r'^(Figure|Chart|Graph|Diagram|Table)\s*\d*\s*[:\.]', re.IGNORECASE)
    note_pattern = re.compile(r'^(Note|Notes|Source|Sources)\s*[:\.]', re.IGNORECASE)
    
    # Паттерн для предложений, заканчивающихся двоеточием (перед формулой)
    formula_intro_pattern = re.compile(r':\s*$')
    
    page_rect = page.rect
    
    for i, block in enumerate(text_blocks):
        # Получаем текст блока
        block_text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "")
        block_text = block_text.strip()
        
        block_bbox = block.get("bbox", (0, 0, 0, 0))
        
        # 1. Проверяем, является ли это подписью фигуры
        if figure_pattern.match(block_text):
            figure_bottom = block_bbox[3]  # y1 - нижняя граница подписи
            
            # Ищем следующий текстовый блок (Note/Source или следующий контент)
            region_bottom = page_rect.height
            
            for j in range(i + 1, len(text_blocks)):
                next_block = text_blocks[j]
                next_bbox = next_block.get("bbox", (0, 0, 0, 0))
                
                next_text = ""
                for line in next_block.get("lines", []):
                    for span in line.get("spans", []):
                        next_text += span.get("text", "")
                next_text = next_text.strip()
                
                if note_pattern.match(next_text):
                    region_bottom = next_bbox[1]
                    break
                
                if figure_pattern.match(next_text):
                    region_bottom = next_bbox[1]
                    break
                
                if next_bbox[1] > figure_bottom + 400:
                    region_bottom = next_bbox[1]
                    break
            
            region_height = region_bottom - figure_bottom
            if region_height > 100:
                regions.append((
                    0, figure_bottom, page_rect.width, region_bottom, figure_bottom
                ))
        
        # 2. Проверяем, заканчивается ли блок двоеточием (перед формулой)
        elif formula_intro_pattern.search(block_text) and i + 1 < len(text_blocks):
            next_block = text_blocks[i + 1]
            next_bbox = next_block.get("bbox", (0, 0, 0, 0))
            
            # Проверяем, что следующий блок близко (формула обычно сразу после)
            gap = next_bbox[1] - block_bbox[3]
            
            if gap < 50:  # Формула должна быть близко к тексту
                next_text = ""
                for line in next_block.get("lines", []):
                    for span in line.get("spans", []):
                        next_text += span.get("text", "")
                next_text = next_text.strip()
                
                # Проверяем, похоже ли это на формулу (короткий текст, спец. символы)
                # Формулы обычно короткие и содержат мат. символы
                has_math_chars = any(c in next_text for c in '=∝∑∫√±×÷αβγδεζηθλμπρσφωΩ∞≈≠≤≥()[]{}')
                is_short = len(next_text) < 200
                
                if has_math_chars or is_short:
                    # Область формулы
                    formula_top = next_bbox[1]
                    formula_bottom = next_bbox[3]
                    
                    # Добавляем небольшой отступ
                    region_height = formula_bottom - formula_top
                    if region_height > 15:  # Минимальная высота формулы
                        regions.append((
                            0, formula_top - 5, page_rect.width, formula_bottom + 5, formula_top
                        ))
    
    return regions


def extract_images_from_page(page: fitz.Page, doc: fitz.Document, 
                              render_drawings: bool = True) -> List[ImageInfo]:
    """
    Извлекает все изображения со страницы PDF.
    Определяет области Figure/Chart и рендерит их как изображения.
    
    Args:
        page: Страница PDF
        doc: Документ PDF
        render_drawings: Рендерить области с графиками
        
    Returns:
        Список ImageInfo с данными изображений
    """
    images = []
    
    # 1. Сначала пытаемся извлечь встроенные растровые изображения
    image_list = page.get_images(full=True)
    
    for img_info in image_list:
        try:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            
            if not base_image:
                continue
                
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png").lower()
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            
            if width < 50 or height < 50:
                continue
            
            if image_ext in ("jpeg", "jpg"):
                image_ext = "jpeg"
            elif image_ext != "png":
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGBA')
                    else:
                        img = img.convert('RGB')
                    output = io.BytesIO()
                    img.save(output, format="PNG")
                    image_bytes = output.getvalue()
                    image_ext = "png"
                except Exception:
                    continue
            
            y_position = 0.0
            for img_rect in page.get_image_rects(xref):
                y_position = img_rect.y0
                break
            
            images.append(ImageInfo(
                data=image_bytes,
                extension=image_ext,
                width=width,
                height=height,
                y_position=y_position
            ))
            
        except Exception:
            continue
    
    # 2. Рендерим области с Figure/Chart как изображения
    if render_drawings:
        try:
            figure_regions = find_figure_regions(page)
            
            for x0, y0, x1, y1, y_pos in figure_regions:
                # Создаём clip rect для области
                clip = fitz.Rect(x0, y0, x1, y1)
                
                # Рендерим только эту область
                zoom = 2.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                
                if pix.width > 50 and pix.height > 50:
                    image_bytes = pix.tobytes("png")
                    
                    images.append(ImageInfo(
                        data=image_bytes,
                        extension="png",
                        width=pix.width,
                        height=pix.height,
                        y_position=y_pos
                    ))
        except Exception:
            pass
    
    return images


def extract_text_blocks(page: fitz.Page) -> List[TextBlock]:
    """
    Извлекает текстовые блоки со страницы и объединяет их в абзацы.
    
    Args:
        page: Страница PDF
        
    Returns:
        Список TextBlock отсортированный по вертикальной позиции
    """
    raw_blocks = []
    
    # Получаем текст в виде словаря с информацией о шрифтах
    page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # Только текстовые блоки
            continue
            
        block_text_parts = []
        block_font_sizes = []
        is_bold = False
        is_italic = False
        
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                text = span.get("text", "")
                line_text += text
                
                if text.strip():
                    block_font_sizes.append(span.get("size", 12))
                    
                    flags = span.get("flags", 0)
                    font_name = span.get("font", "").lower()
                    
                    if (flags & 16) or "bold" in font_name:
                        is_bold = True
                    if (flags & 2) or "italic" in font_name or "oblique" in font_name:
                        is_italic = True
            
            # Добавляем строку, удаляя лишние пробелы в конце
            line_text = line_text.rstrip()
            if line_text:
                block_text_parts.append(line_text)
        
        if not block_text_parts:
            continue
        
        # Объединяем строки в текст блока
        # Если строка заканчивается дефисом, убираем его (перенос слова)
        full_text_parts = []
        for i, part in enumerate(block_text_parts):
            if part.endswith('-') and i < len(block_text_parts) - 1:
                # Перенос слова - убираем дефис и не добавляем пробел
                full_text_parts.append(part[:-1])
            else:
                full_text_parts.append(part + ' ')
        
        full_text = ''.join(full_text_parts).strip()
        full_text = re.sub(r'\s+', ' ', full_text)
        
        if not full_text:
            continue
        
        avg_font_size = sum(block_font_sizes) / len(block_font_sizes) if block_font_sizes else 12
        bbox = block.get("bbox", (0, 0, 0, 0))
        
        raw_blocks.append(TextBlock(
            text=full_text,
            x0=bbox[0],
            y0=bbox[1],
            x1=bbox[2],
            y1=bbox[3],
            font_size=avg_font_size,
            is_bold=is_bold,
            is_italic=is_italic
        ))
    
    # Сортируем блоки по вертикальной позиции
    raw_blocks.sort(key=lambda b: (b.y0, b.x0))
    
    # Объединяем близкие блоки в абзацы
    merged_blocks = []
    
    for block in raw_blocks:
        if not merged_blocks:
            merged_blocks.append(block)
            continue
        
        last_block = merged_blocks[-1]
        
        # Определяем, нужно ли объединять с предыдущим блоком
        # Условия для объединения:
        # 1. Блоки близки по вертикали (gap < 1.5 * font_size)
        # 2. Одинаковый размер шрифта (±20%)
        # 3. Предыдущий блок не заканчивается точкой/вопросом/восклицанием + новый с заглавной
        
        vertical_gap = block.y0 - last_block.y1
        font_size_ratio = block.font_size / last_block.font_size if last_block.font_size > 0 else 1
        
        # Горизонтальный отступ (indent) - признак нового абзаца
        horizontal_indent = block.x0 - last_block.x0
        has_indent = horizontal_indent > 10  # Отступ больше 10pt = новый абзац
        
        same_paragraph = (
            vertical_gap < last_block.font_size * 1.5 and
            0.8 < font_size_ratio < 1.2 and
            not last_block.is_bold != block.is_bold and  # Оба одного стиля
            not has_indent  # Нет отступа = тот же абзац
        )
        
        # Не объединяем если предыдущий закончился предложением и новый с заглавной
        if same_paragraph:
            last_text = last_block.text.rstrip()
            ends_sentence = last_text and last_text[-1] in '.!?'
            starts_capital = block.text and block.text[0].isupper()
            
            # Если заканчивается предложение И начинается с заглавной И есть вертикальный отступ
            # - это новый абзац
            if ends_sentence and starts_capital and vertical_gap > last_block.font_size * 0.8:
                same_paragraph = False
        
        if same_paragraph:
            # Объединяем блоки
            combined_text = last_block.text
            # Добавляем пробел между блоками
            if not combined_text.endswith(' ') and not block.text.startswith(' '):
                combined_text += ' '
            combined_text += block.text
            
            merged_blocks[-1] = TextBlock(
                text=combined_text,
                x0=min(last_block.x0, block.x0),
                y0=last_block.y0,
                x1=max(last_block.x1, block.x1),
                y1=block.y1,
                font_size=(last_block.font_size + block.font_size) / 2,
                is_bold=last_block.is_bold or block.is_bold,
                is_italic=last_block.is_italic or block.is_italic
            )
        else:
            merged_blocks.append(block)
    
    return merged_blocks


def extract_page_content(page: fitz.Page, doc: fitz.Document, page_num: int) -> PageContent:
    """
    Извлекает всё содержимое страницы: текст и изображения.
    Текст из областей Figure исключается (он будет на рендеренной картинке).
    
    Args:
        page: Страница PDF
        doc: Документ PDF
        page_num: Номер страницы
        
    Returns:
        PageContent с текстом и изображениями
    """
    # Сначала находим области Figure
    figure_regions = find_figure_regions(page)
    
    # Извлекаем все текстовые блоки
    text_blocks = extract_text_blocks(page)
    
    # Фильтруем текстовые блоки — убираем те, что внутри областей Figure
    filtered_blocks = []
    for block in text_blocks:
        block_inside_figure = False
        
        for x0, y0, x1, y1, _ in figure_regions:
            # Проверяем, находится ли центр блока внутри области Figure
            block_center_y = (block.y0 + block.y1) / 2
            if y0 < block_center_y < y1:
                block_inside_figure = True
                break
        
        if not block_inside_figure:
            filtered_blocks.append(block)
    
    # Извлекаем изображения
    images = extract_images_from_page(page, doc)
    
    return PageContent(
        page_num=page_num,
        text_blocks=filtered_blocks,
        images=images
    )


def escape_html(text: str) -> str:
    """Экранирует специальные HTML символы."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    return text


def calculate_median_font_size(pages: List[PageContent]) -> float:
    """Вычисляет медианный размер шрифта в документе."""
    sizes = []
    for page in pages:
        for block in page.text_blocks:
            sizes.append(block.font_size)
    
    if not sizes:
        return 12.0
    
    sizes.sort()
    mid = len(sizes) // 2
    return sizes[mid] if len(sizes) % 2 == 1 else (sizes[mid - 1] + sizes[mid]) / 2


def is_heading(block: TextBlock, median_font_size: float) -> Optional[int]:
    """
    Определяет, является ли блок заголовком и какого уровня.
    
    Returns:
        Уровень заголовка (1-3) или None
    """
    ratio = block.font_size / median_font_size
    
    # Заголовки обычно полужирные или крупнее обычного текста
    if ratio > 1.5 or (ratio > 1.2 and block.is_bold):
        if ratio > 1.8:
            return 1
        elif ratio > 1.4:
            return 2
        else:
            return 3
    
    return None


def page_content_to_html(page: PageContent, median_font_size: float, 
                         image_items: List[Tuple[str, epub.EpubItem]]) -> str:
    """
    Конвертирует содержимое страницы в HTML.
    
    Args:
        page: Содержимое страницы
        median_font_size: Медианный размер шрифта
        image_items: Список для добавления изображений (filename, EpubItem)
        
    Returns:
        HTML строка
    """
    html_parts = []
    
    # Собираем все элементы (текст и изображения) с их позициями
    elements = []
    
    for block in page.text_blocks:
        elements.append(("text", block.y0, block))
    
    for img in page.images:
        elements.append(("image", img.y_position, img))
    
    # Сортируем по вертикальной позиции
    elements.sort(key=lambda x: x[1])
    
    for elem_type, _, elem in elements:
        if elem_type == "text":
            block = elem
            text = escape_html(block.text)
            
            heading_level = is_heading(block, median_font_size)
            
            if heading_level:
                html_parts.append(f"<h{heading_level}>{text}</h{heading_level}>")
            else:
                # Применяем форматирование
                if block.is_bold and block.is_italic:
                    text = f"<strong><em>{text}</em></strong>"
                elif block.is_bold:
                    text = f"<strong>{text}</strong>"
                elif block.is_italic:
                    text = f"<em>{text}</em>"
                
                html_parts.append(f"<p>{text}</p>")
        
        elif elem_type == "image":
            img = elem
            # Имя файла будет добавлено позже при обработке главы
            img_index = len(image_items)
            img_filename = f"images/img_{page.page_num}_{img_index}.{img.extension}"
            
            # Создаём EpubItem для изображения
            img_item = epub.EpubItem(
                uid=f"img_{page.page_num}_{img_index}",
                file_name=img_filename,
                media_type=f"image/{img.extension}",
                content=img.data
            )
            image_items.append((img_filename, img_item))
            
            html_parts.append(f'<div class="image-container"><img src="{img_filename}" alt=""/></div>')
    
    return "\n".join(html_parts)


def create_chapter(chapter_idx: int, pages: List[PageContent], 
                   median_font_size: float, book: epub.EpubBook, css) -> epub.EpubHtml:
    """
    Создаёт главу EPUB из списка страниц.
    
    Args:
        chapter_idx: Индекс главы
        pages: Список страниц для этой главы
        median_font_size: Медианный размер шрифта
        book: Объект книги EPUB
        css: CSS стили
        
    Returns:
        EpubHtml глава
    """
    image_items = []
    html_parts = []
    
    for page in pages:
        page_html = page_content_to_html(page, median_font_size, image_items)
        if page_html.strip():
            html_parts.append(page_html)
            html_parts.append('<hr class="page-break"/>')
    
    # Удаляем последний разделитель страниц
    if html_parts and html_parts[-1] == '<hr class="page-break"/>':
        html_parts.pop()
    
    chapter_html = "\n".join(html_parts)
    
    # Определяем название главы
    first_page = pages[0].page_num + 1
    last_page = pages[-1].page_num + 1
    
    if first_page == last_page:
        chapter_title = f"Страница {first_page}"
    else:
        chapter_title = f"Страницы {first_page}-{last_page}"
    
    # Пытаемся найти заголовок в первом блоке
    for page in pages:
        for block in page.text_blocks:
            if is_heading(block, median_font_size):
                # Используем текст заголовка как название главы
                title_text = block.text[:60]
                if len(block.text) > 60:
                    title_text += "..."
                chapter_title = title_text
                break
        else:
            continue
        break
    
    # Гарантируем непустой контент
    if not chapter_html.strip():
        chapter_html = f"<p>{escape_html(chapter_title)}</p>"
    
    # Добавляем изображения в книгу
    for _, img_item in image_items:
        book.add_item(img_item)
    
    # Создаём главу
    chapter = epub.EpubHtml(
        title=chapter_title,
        file_name=f"chapter_{chapter_idx + 1}.xhtml",
        lang="ru"
    )
    
    # Формируем полный HTML документ
    full_html = f'''<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{escape_html(chapter_title)}</title>
    <link rel="stylesheet" type="text/css" href="style/main.css"/>
</head>
<body>
<h1 class="chapter-title">{escape_html(chapter_title)}</h1>
{chapter_html}
</body>
</html>'''
    
    chapter.content = full_html.encode('utf-8')
    chapter.add_item(css)
    
    return chapter


def create_epub_from_pdf(pdf_path: str, epub_path: str,
                         title: Optional[str] = None,
                         author: Optional[str] = None) -> bool:
    """
    Создаёт EPUB файл из PDF.
    
    Args:
        pdf_path: Путь к PDF файлу
        epub_path: Путь к выходному EPUB файлу
        title: Название книги
        author: Автор книги
        
    Returns:
        True при успешной конвертации
    """
    try:
        print(f"Открываю PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        
        # Получаем метаданные
        metadata = doc.metadata or {}
        if not title:
            title = metadata.get("title") or Path(pdf_path).stem
        if not author:
            author = metadata.get("author") or "Unknown"
        
        print(f"  Название: {title}")
        print(f"  Автор: {author}")
        print(f"  Страниц: {len(doc)}")
        
        # Извлекаем содержимое всех страниц
        print("  Извлекаю содержимое страниц...")
        pages_content = []
        total_images = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            content = extract_page_content(page, doc, page_num)
            pages_content.append(content)
            total_images += len(content.images)
            
            if (page_num + 1) % 10 == 0:
                print(f"    Обработано страниц: {page_num + 1}/{len(doc)}")
        
        print(f"  Извлечено изображений: {total_images}")
        
        # Вычисляем медианный размер шрифта
        median_font_size = calculate_median_font_size(pages_content)
        print(f"  Медианный размер шрифта: {median_font_size:.1f}")
        
        # Создаём EPUB
        print("  Создаю EPUB...")
        book = epub.EpubBook()
        
        # Устанавливаем метаданные
        book.set_identifier(f"pdf2epub_{Path(pdf_path).stem}")
        book.set_title(title)
        book.set_language("ru")
        book.add_author(author)
        
        # Добавляем CSS стили
        style = '''
body {
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.6;
    margin: 1em;
    text-align: justify;
}
h1, h2, h3 {
    font-family: Arial, Helvetica, sans-serif;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    text-align: left;
}
h1 { font-size: 1.8em; }
h2 { font-size: 1.5em; }
h3 { font-size: 1.2em; }
h1.chapter-title {
    font-size: 1.5em;
    text-align: center;
    margin-bottom: 2em;
    color: #333;
}
p {
    text-indent: 1.5em;
    margin: 0.5em 0;
}
strong { font-weight: bold; }
em { font-style: italic; }
.image-container {
    text-align: center;
    margin: 1.5em 0;
}
.image-container img {
    max-width: 100%;
    height: auto;
}
hr.page-break {
    border: none;
    border-top: 1px solid #ccc;
    margin: 2em 0;
}
'''
        
        css = epub.EpubItem(
            uid="style",
            file_name="style/main.css",
            media_type="text/css",
            content=style.encode('utf-8')
        )
        book.add_item(css)
        
        # Разбиваем на главы
        chapters = []
        num_chapters = (len(pages_content) + PAGES_PER_CHAPTER - 1) // PAGES_PER_CHAPTER
        
        print(f"  Создаю {num_chapters} глав(ы)...")
        
        for chapter_idx in range(num_chapters):
            start_page = chapter_idx * PAGES_PER_CHAPTER
            end_page = min(start_page + PAGES_PER_CHAPTER, len(pages_content))
            
            chapter_pages = pages_content[start_page:end_page]
            chapter = create_chapter(chapter_idx, chapter_pages, median_font_size, book, css)
            
            book.add_item(chapter)
            chapters.append(chapter)
        
        # Создаём оглавление
        book.toc = chapters
        
        # Добавляем NCX и Nav
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # Устанавливаем spine
        book.spine = ["nav"] + chapters
        
        # Создаём выходную директорию если нужно
        output_dir = Path(epub_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем EPUB
        print(f"  Сохраняю EPUB: {epub_path}")
        epub.write_epub(epub_path, book, {})
        
        doc.close()
        print(f"  [OK] Конвертация завершена!")
        print(f"    Создано глав: {len(chapters)}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Ошибка при конвертации: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_folder(input_folder: str, output_folder: str) -> Tuple[int, int]:
    """
    Обрабатывает папку с PDF файлами.
    
    Args:
        input_folder: Путь к входной папке
        output_folder: Путь к выходной папке
        
    Returns:
        Кортеж (успешно, всего)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"Ошибка: Папка не существует: {input_folder}")
        return 0, 0
    
    # Создаём выходную папку
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Находим PDF файлы
    pdf_files = list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF"))
    # Убираем дубликаты
    pdf_files = list(set(pdf_files))
    
    if not pdf_files:
        print(f"PDF файлы не найдены в: {input_folder}")
        return 0, 0
    
    print(f"Найдено PDF файлов: {len(pdf_files)}")
    print("-" * 50)
    
    success_count = 0
    
    for pdf_file in pdf_files:
        epub_name = pdf_file.stem + ".epub"
        epub_file = output_path / epub_name
        
        if create_epub_from_pdf(str(pdf_file), str(epub_file)):
            success_count += 1
        
        print("-" * 50)
    
    return success_count, len(pdf_files)


def main():
    parser = argparse.ArgumentParser(
        description="Конвертер PDF в EPUB с улучшенным извлечением текста и изображений",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                              Конвертировать все PDF из стандартной папки
  %(prog)s book.pdf                     Конвертировать book.pdf в book.epub
  %(prog)s book.pdf -o mybook.epub      Указать имя выходного файла
  %(prog)s ./pdfs -o ./epubs            Конвертировать все PDF из папки
  %(prog)s book.pdf -t "Моя книга"      Указать название книги
  %(prog)s book.pdf -a "Иван Иванов"    Указать автора
        """
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=f"Путь к PDF файлу или папке с PDF файлами (по умолчанию: {DEFAULT_INPUT_PATH})"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Путь к выходному EPUB файлу или папке"
    )
    
    parser.add_argument(
        "-t", "--title",
        help="Название книги"
    )
    
    parser.add_argument(
        "-a", "--author",
        help="Автор книги"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Ошибка: Путь не существует: {args.input}")
        sys.exit(1)
    
    if input_path.is_dir():
        # Обрабатываем папку
        output_folder = args.output or str(input_path) + "_epub"
        success, total = process_folder(args.input, output_folder)
        
        print("=" * 50)
        print(f"Результат: {success}/{total} файлов успешно конвертировано")
        
        if success < total:
            sys.exit(1)
    else:
        # Обрабатываем один файл
        if args.output:
            output_path = args.output
        else:
            output_path = str(input_path.with_suffix(".epub"))
        
        if not create_epub_from_pdf(args.input, output_path, args.title, args.author):
            sys.exit(1)


if __name__ == "__main__":
    main()
