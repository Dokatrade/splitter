"""
Скрипт для разделения EPUB книги на две части.
Разделяет HTML-контент внутри глав и распределяет изображения.
"""

import os
import zipfile
import shutil
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import re
from collections import defaultdict


def split_epub_advanced(epub_path: str, output_dir: str = None):
    """
    Разделяет EPUB файл на две примерно равные по размеру части.
    Если глав мало, разделяет контент внутри HTML.
    """
    epub_path = Path(epub_path)
    
    if output_dir is None:
        output_dir = epub_path.parent
    else:
        output_dir = Path(output_dir)
    
    base_name = epub_path.stem
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print(f"Распаковка {epub_path.name}...")
        with zipfile.ZipFile(epub_path, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
        
        # Собираем информацию об изображениях
        images_dir = None
        all_images = {}
        for img_path in temp_path.rglob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']:
                all_images[img_path.name] = {
                    'path': img_path,
                    'size': img_path.stat().st_size
                }
                if images_dir is None:
                    images_dir = img_path.parent
        
        total_img_size = sum(img['size'] for img in all_images.values())
        print(f"Найдено {len(all_images)} изображений, общий размер: {total_img_size/(1024*1024):.2f} МБ")
        
        # Находим главный HTML
        opf_file = find_opf_file(temp_path)
        opf_dir = opf_file.parent
        
        tree = ET.parse(opf_file)
        root = tree.getroot()
        ns = get_namespaces(root)
        manifest = get_manifest_items(root, ns, opf_dir)
        spine_items = get_spine_items(root, ns, manifest)
        
        # Находим главный HTML с изображениями
        main_html_path = None
        main_html_id = None
        for item_id in spine_items:
            if item_id in manifest:
                item_path = manifest[item_id]['path']
                if item_path.exists() and item_path.stat().st_size > 100000:  # > 100KB
                    main_html_path = item_path
                    main_html_id = item_id
                    break
        
        if not main_html_path:
            # Берем самый большой HTML
            max_size = 0
            for item_id in spine_items:
                if item_id in manifest:
                    item_path = manifest[item_id]['path']
                    if item_path.exists() and item_path.stat().st_size > max_size:
                        max_size = item_path.stat().st_size
                        main_html_path = item_path
                        main_html_id = item_id
        
        print(f"Главный HTML: {main_html_path.name}")
        
        # Читаем HTML
        with open(main_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Находим все ссылки на изображения и их позиции
        img_refs = list(re.finditer(r'(?:src|href)=["\']([^"\']+\.(?:jpg|jpeg|png|gif|webp|svg))["\']', 
                                    html_content, re.IGNORECASE))
        
        print(f"Найдено {len(img_refs)} ссылок на изображения в HTML")
        
        # Вычисляем размер изображений до каждой позиции
        cumulative_sizes = []
        current_size = 0
        for match in img_refs:
            img_name = Path(match.group(1)).name
            if img_name in all_images:
                current_size += all_images[img_name]['size']
            cumulative_sizes.append((match.start(), current_size, img_name))
        
        # Находим точку разделения (примерно 50% по размеру изображений)
        target_size = total_img_size / 2
        split_pos = 0
        
        for pos, cum_size, img_name in cumulative_sizes:
            if cum_size >= target_size:
                split_pos = pos
                break
        
        if split_pos == 0:
            split_pos = len(html_content) // 2
        
        # Ищем хорошее место для разделения (заголовок или параграф)
        search_zone = html_content[max(0, split_pos - 2000):min(len(html_content), split_pos + 2000)]
        
        # Ищем заголовки h1-h3 или теги p
        split_patterns = [
            (r'<h[1-3][^>]*>', 'heading'),
            (r'<p[^>]*>', 'paragraph'),
            (r'<div[^>]*class="[^"]*chapter[^"]*"[^>]*>', 'chapter'),
        ]
        
        best_split = split_pos
        for pattern, name in split_patterns:
            matches = list(re.finditer(pattern, search_zone, re.IGNORECASE))
            if matches:
                # Берем ближайший к середине зоны
                mid = len(search_zone) // 2
                best_match = min(matches, key=lambda m: abs(m.start() - mid))
                best_split = max(0, split_pos - 2000) + best_match.start()
                print(f"Найдена точка разделения: {name}")
                break
        
        split_pos = best_split
        
        # Подсчитываем изображения в каждой части
        part1_images = set()
        part2_images = set()
        
        for match in img_refs:
            img_name = Path(match.group(1)).name
            if match.start() < split_pos:
                part1_images.add(img_name)
            else:
                part2_images.add(img_name)
        
        size1 = sum(all_images[name]['size'] for name in part1_images if name in all_images)
        size2 = sum(all_images[name]['size'] for name in part2_images if name in all_images)
        
        print(f"\nРазделение:")
        print(f"  Часть 1: {len(part1_images)} изображений, ~{size1/(1024*1024):.2f} МБ")
        print(f"  Часть 2: {len(part2_images)} изображений, ~{size2/(1024*1024):.2f} МБ")
        
        # Создаем две части HTML
        html_part1 = html_content[:split_pos]
        html_part2 = html_content[split_pos:]
        
        # Добавляем закрывающие теги к первой части и открывающие ко второй
        # Находим head и начало body
        head_match = re.search(r'(<head[^>]*>.*?</head>)', html_content, re.DOTALL | re.IGNORECASE)
        body_start_match = re.search(r'(<body[^>]*>)', html_content, re.IGNORECASE)
        html_start_match = re.search(r'(<\?xml[^>]*\?>)?\s*(<html[^>]*>)', html_content, re.IGNORECASE)
        
        head_content = head_match.group(1) if head_match else '<head><title>Part</title></head>'
        html_start = html_start_match.group(0) if html_start_match else '<?xml version="1.0" encoding="utf-8"?>\n<html xmlns="http://www.w3.org/1999/xhtml">'
        body_start = body_start_match.group(1) if body_start_match else '<body>'
        
        # Формируем полные HTML части
        html_part1_full = html_part1
        if not html_part1.strip().endswith('</body>'):
            html_part1_full += '\n</body>\n</html>'
        
        html_part2_full = f'{html_start}\n{head_content}\n{body_start}\n{html_part2}'
        if not html_part2_full.strip().endswith('</html>'):
            pass  # Должен уже содержать закрывающие теги
        
        # Создаем части EPUB
        part1_path = output_dir / f"{base_name}_part1.epub"
        part2_path = output_dir / f"{base_name}_part2.epub"
        
        print(f"\nСоздание части 1...")
        create_epub_from_split(temp_path, opf_file, main_html_path, html_part1_full, 
                               part1_images, all_images, ns, manifest, spine_items, part1_path, 1)
        
        print(f"Создание части 2...")
        create_epub_from_split(temp_path, opf_file, main_html_path, html_part2_full, 
                               part2_images, all_images, ns, manifest, spine_items, part2_path, 2)
        
        print(f"\nГотово!")
        print(f"  Часть 1: {part1_path}")
        print(f"  Часть 2: {part2_path}")
        
        size1_actual = part1_path.stat().st_size / (1024 * 1024)
        size2_actual = part2_path.stat().st_size / (1024 * 1024)
        original_size = epub_path.stat().st_size / (1024 * 1024)
        
        print(f"\nРазмеры файлов:")
        print(f"  Оригинал: {original_size:.2f} МБ")
        print(f"  Часть 1:  {size1_actual:.2f} МБ")
        print(f"  Часть 2:  {size2_actual:.2f} МБ")


def create_epub_from_split(original_dir: Path, opf_file: Path, main_html: Path, 
                           html_content: str, images_to_include: set, all_images: dict,
                           ns: dict, manifest: dict, spine_items: list, 
                           output_path: Path, part_num: int):
    """Создает EPUB из разделенного HTML."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Копируем META-INF
        meta_inf_src = original_dir / "META-INF"
        if meta_inf_src.exists():
            shutil.copytree(meta_inf_src, temp_path / "META-INF")
        
        # Копируем mimetype
        mimetype_src = original_dir / "mimetype"
        if mimetype_src.exists():
            shutil.copy2(mimetype_src, temp_path / "mimetype")
        
        # Создаем структуру OEBPS
        oebps_src = original_dir / "OEBPS"
        oebps_dst = temp_path / "OEBPS"
        
        # Копируем CSS
        if (oebps_src / "Styles").exists():
            shutil.copytree(oebps_src / "Styles", oebps_dst / "Styles")
        
        # Копируем нужные изображения
        images_dst = oebps_dst / "Images"
        images_dst.mkdir(parents=True, exist_ok=True)
        
        for img_name in images_to_include:
            if img_name in all_images:
                src = all_images[img_name]['path']
                dst = images_dst / img_name
                shutil.copy2(src, dst)
        
        # Также копируем обложку если есть
        for img_name, img_info in all_images.items():
            if 'cover' in img_name.lower():
                dst = images_dst / img_name
                if not dst.exists():
                    shutil.copy2(img_info['path'], dst)
        
        # Сохраняем HTML
        text_dst = oebps_dst / "Text"
        text_dst.mkdir(parents=True, exist_ok=True)
        
        main_html_dst = text_dst / main_html.name
        with open(main_html_dst, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Копируем другие HTML (кроме главного)
        if (oebps_src / "Text").exists():
            for html_file in (oebps_src / "Text").glob("*.htm*"):
                if html_file.name != main_html.name:
                    shutil.copy2(html_file, text_dst / html_file.name)
        
        # Создаем OPF
        new_opf = oebps_dst / "content.opf"
        create_split_opf(opf_file, new_opf, images_to_include, ns, part_num)
        
        # Копируем NCX если есть
        for ncx in original_dir.rglob("*.ncx"):
            shutil.copy2(ncx, oebps_dst / ncx.name)
        
        # Создаем EPUB
        create_epub_archive(temp_path, output_path)


def create_split_opf(original_opf: Path, new_opf: Path, images_to_include: set, 
                     ns: dict, part_num: int):
    """Создает OPF для части."""
    tree = ET.parse(original_opf)
    root = tree.getroot()
    
    # Обновляем title
    title_elem = root.find('.//{%s}title' % ns['dc'])
    if title_elem is not None and title_elem.text:
        title_elem.text = f"{title_elem.text} (Chast {part_num})"
    
    # Удаляем ненужные изображения из manifest
    manifest_elem = root.find('.//{%s}manifest' % ns['opf'])
    if manifest_elem is not None:
        items_to_remove = []
        for item in manifest_elem.findall('{%s}item' % ns['opf']):
            href = item.get('href', '')
            media_type = item.get('media-type', '')
            
            # Если это изображение, проверяем нужно ли оно
            if 'image' in media_type:
                img_name = Path(href).name
                if img_name not in images_to_include and 'cover' not in img_name.lower():
                    items_to_remove.append(item)
        
        for item in items_to_remove:
            manifest_elem.remove(item)
    
    new_opf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(new_opf, encoding='utf-8', xml_declaration=True)


def find_opf_file(epub_dir: Path) -> Path:
    container_path = epub_dir / "META-INF" / "container.xml"
    if container_path.exists():
        tree = ET.parse(container_path)
        root = tree.getroot()
        ns = {'container': 'urn:oasis:names:tc:opendocument:xmlns:container'}
        rootfile = root.find('.//container:rootfile', ns)
        if rootfile is not None:
            opf_path = rootfile.get('full-path')
            if opf_path:
                return epub_dir / opf_path
    
    for opf in epub_dir.rglob('*.opf'):
        return opf
    return None


def get_namespaces(root) -> dict:
    ns = {}
    tag = root.tag
    if tag.startswith('{'):
        ns['opf'] = tag[1:tag.index('}')]
    else:
        ns['opf'] = 'http://www.idpf.org/2007/opf'
    ns['dc'] = 'http://purl.org/dc/elements/1.1/'
    return ns


def get_manifest_items(root, ns: dict, opf_dir: Path) -> dict:
    manifest = {}
    manifest_elem = root.find('.//{%s}manifest' % ns['opf'])
    if manifest_elem is not None:
        for item in manifest_elem.findall('{%s}item' % ns['opf']):
            item_id = item.get('id')
            href = item.get('href')
            media_type = item.get('media-type')
            manifest[item_id] = {
                'href': href,
                'media_type': media_type,
                'path': opf_dir / href
            }
    return manifest


def get_spine_items(root, ns: dict, manifest: dict) -> list:
    spine_items = []
    spine_elem = root.find('.//{%s}spine' % ns['opf'])
    if spine_elem is not None:
        for itemref in spine_elem.findall('{%s}itemref' % ns['opf']):
            idref = itemref.get('idref')
            if idref in manifest:
                spine_items.append(idref)
    return spine_items


def create_epub_archive(source_dir: Path, output_path: Path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        mimetype_path = source_dir / "mimetype"
        if mimetype_path.exists():
            zipf.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file == "mimetype":
                    continue
                file_path = Path(root) / file
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)


def main():
    epub_folder = Path(r"d:\results\epub_razdel")
    
    # Удаляем старые части
    for old_part in epub_folder.glob("*_part*.epub"):
        old_part.unlink()
        print(f"Удален: {old_part.name}")
    
    epub_files = list(epub_folder.glob("*.epub"))
    epub_files = [f for f in epub_files if "_part" not in f.name]
    
    if not epub_files:
        print("Не найдены EPUB файлы")
        return
    
    epub_file = epub_files[0]
    print(f"Файл: {epub_file.name}")
    print(f"Размер: {epub_file.stat().st_size / (1024 * 1024):.2f} МБ\n")
    
    split_epub_advanced(epub_file, epub_folder)


if __name__ == "__main__":
    main()
