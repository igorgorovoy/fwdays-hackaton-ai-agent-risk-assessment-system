from PIL import Image
import os

def create_reversed_cards():
    # Отримуємо шлях до поточної директорії скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Папки з картами
    subfolders = ['MajorArcana', 'MinorArcana_Cups', 'MinorArcana_Pentacles', 
                  'MinorArcana_Swords', 'MinorArcana_Wands']
    
    # Координати області з текстом (припустимо, що текст знаходиться внизу карти)
    # Ці значення потрібно налаштувати відповідно до ваших карт
    TEXT_TOP = 800    # Верхня межа тексту
    TEXT_BOTTOM = 900 # Нижня межа тексту
    
    for folder in subfolders:
        # Використовуємо абсолютний шлях відносно директорії скрипта
        folder_path = os.path.join(script_dir, 'cards', folder)
        
        # Перевіряємо чи існує директорія
        if not os.path.exists(folder_path):
            print(f"Директорія не існує: {folder_path}")
            continue
            
        # Отримуємо всі файли зображень
        card_files = [file for file in os.listdir(folder_path) 
                     if file.endswith(('.jpg', '.png'))]
        
        for card_file in card_files:
            # Пропускаємо файли, які вже є перевернутими
            if card_file.endswith('-r.jpg') or card_file.endswith('-r.png'):
                continue
                
            input_path = os.path.join(folder_path, card_file)
            
            # Створюємо ім'я для перевернутого файлу
            filename, ext = os.path.splitext(card_file)
            reversed_filename = f"{filename}-r{ext}"
            output_path = os.path.join(folder_path, reversed_filename)
            
            try:
                # Відкриваємо зображення
                with Image.open(input_path) as img:
                    # Створюємо копію оригінального зображення
                    reversed_img = img.copy()
                    
                    # Отримуємо розміри зображення
                    width, height = img.size
                    
                    # Створюємо маску для тексту
                    text_region = img.crop((0, TEXT_TOP, width, TEXT_BOTTOM))
                    
                    # Перевертаємо все зображення
                    reversed_img = reversed_img.rotate(180)
                    
                    # Вставляємо оригінальний текст назад
                    reversed_img.paste(text_region, (0, height - TEXT_BOTTOM))
                    
                    # Зберігаємо результат
                    reversed_img.save(output_path)
                print(f"Створено перевернуту версію для {card_file}")
            except Exception as e:
                print(f"Помилка при обробці {card_file}: {str(e)}")

if __name__ == "__main__":
    create_reversed_cards()