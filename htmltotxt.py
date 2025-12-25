import os
import re
from bs4 import BeautifulSoup

# НАСТРОЙКИ
# Путь к папке с экспортированным чатом Telegram.
# Оставьте пустым (""), если скрипт лежит в той же папке, что и HTML-файлы.
CHAT_FOLDER = "d:/results/Affiliatka&aishka to txt/ChatExport_2025-12-24"  # Например: "D:/Telegram/ChatExport_2024-01-01"

BASE_FILENAME = "chat_part"
# Папка для сохранения результатов. Будет создана автоматически.
OUTPUT_FOLDER = "d:/results/Affiliatka&aishka to txt/"
# Ставим жесткий лимит - 50 000 слов на файл. 
# Это создаст больше файлов, но они гарантированно загрузятся.
WORDS_PER_FILE_LIMIT = 50000 

def get_file_number(filename):
    if filename == "messages.html":
        return 1
    match = re.search(r'messages(\d+)\.html', filename)
    return int(match.group(1)) if match else 0

def clean_text(text):
    return text.strip() if text else ""

def count_words(text):
    # Грубый подсчет слов (по пробелам)
    return len(text.split())

def main():
    # Определяем рабочую директорию
    work_dir = CHAT_FOLDER if CHAT_FOLDER else '.'
    
    # Создаем папку для результатов, если её нет
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Создана папка для результатов: {OUTPUT_FOLDER}")
    
    files = [f for f in os.listdir(work_dir) if f.startswith('messages') and f.endswith('.html')]
    files.sort(key=get_file_number)

    print(f"Найдено файлов HTML: {len(files)}.")
    
    current_part = 1
    current_word_count = 0
    
    # Создаем первый файл в папке результатов
    current_file_name = os.path.join(OUTPUT_FOLDER, f"{BASE_FILENAME}_{current_part}.txt")
    outfile = open(current_file_name, 'w', encoding='utf-8-sig')
    
    print(f"Пишем в {current_file_name}...")

    for file_name in files:
        file_path = os.path.join(work_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as html_file:
                soup = BeautifulSoup(html_file, 'html.parser')
                messages = soup.find_all('div', class_='message')
                
                for msg in messages:
                    if not msg.find('div', class_='body'):
                        continue

                    # Данные
                    date_div = msg.find('div', class_='pull_right date details')
                    date = date_div['title'] if date_div and date_div.has_attr('title') else "?"
                    
                    from_name_div = msg.find('div', class_='from_name')
                    sender = clean_text(from_name_div.text) if from_name_div else "System"
                    
                    text_div = msg.find('div', class_='text')
                    if text_div:
                        text = clean_text(text_div.get_text(separator=" "))
                        entry_words = count_words(text)
                        
                        # ПРОВЕРКА: Если добавив это сообщение, мы превысим лимит -> режем
                        if current_word_count + entry_words > WORDS_PER_FILE_LIMIT:
                            outfile.close()
                            print(f"--- Файл {current_file_name} готов. Слов: {current_word_count}")
                            
                            current_part += 1
                            current_word_count = 0
                            current_file_name = os.path.join(OUTPUT_FOLDER, f"{BASE_FILENAME}_{current_part}.txt")
                            outfile = open(current_file_name, 'w', encoding='utf-8-sig')
                            print(f"Пишем в {current_file_name}...")
                        
                        outfile.write(f"[{date}] {sender}: {text}\n")
                        outfile.write("-" * 20 + "\n")
                        current_word_count += entry_words

        except Exception as e:
            print(f"Ошибка в файле {file_name}: {e}")

    outfile.close()
    print(f"\nГотово! Чат разбит на {current_part} частей. Загрузите их все в NotebookLM.")

if __name__ == "__main__":
    main()