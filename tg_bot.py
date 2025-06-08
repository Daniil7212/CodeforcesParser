import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import csv
import pyperclip
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from telebot import *
import telebot
import my_ai

# Настройки Telegram бота
TELEGRAM_BOT_TOKEN = '5560616880:AAGglfnaBXoft3gUC_tJ40Vuf7iL2-C-DlE'
TELEGRAM_CHAT_ID = '2112133119'  # ID чата, куда будут отправляться логи
flag = False
# Инициализация бота
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Глобальные переменные для хранения состояния
user_data = {}

classifier = my_ai.create_model(data_path="code_dataset.csv", model_type="nn", epochs=100)

def send_log(message: str):
    """Отправляет сообщение в Telegram."""
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        print(f"Ошибка при отправке сообщения в Telegram: {e}")

def parser_wrapper(chat_id):
    """Запускает парсер с данными из user_data"""
    data = user_data.get(chat_id)
    if not data:
        bot.send_message(chat_id, "Ошибка: данные не найдены")
        return
    
    with open("deepseek_data.json", "r", encoding="utf-8") as fle:
        json_data = json.load(fle)
    json_data = json_data["students"]
    
    if data['surname'] == "-":
        for student in json_data:
            for contest_id in data['ids']:
                parser(contest_id, True, student["login"], student["password"])
    else:
        for student in json_data:
            if data['name'] == student["name"] and data['surname'] == student["surname"]:
                for contest_id in data['ids']:
                    parser(contest_id, True, student["login"], student["password"])
                break

@bot.message_handler(commands=['start'])
def start(message):
    """Обработчик команды /start"""
    global TELEGRAM_CHAT_ID
    global flag
    TELEGRAM_CHAT_ID = message.chat.id
    print("Пришел запрос от ", "@" + message.from_user.username)
    if message.from_user.username == "daniil7217":
        flag = True
    print("Имя пользователя: ", message.from_user.first_name)
    user_data[TELEGRAM_CHAT_ID] = {}
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(types.KeyboardButton("Все пользователи"))
    bot.send_message(TELEGRAM_CHAT_ID, 
                    "Введите фамилию (или нажмите 'Все пользователи'):", 
                    reply_markup=markup)

@bot.message_handler(func=lambda message: 'surname' not in user_data.get(message.chat.id, {}))
def get_surname(message):
    """Получаем фамилию пользователя"""
    chat_id = message.chat.id
    user_data[chat_id]['surname'] = message.text
    if message.text != "Все пользователи":
        surname = message.text
        bot.send_message(chat_id, "Введите имя:", reply_markup=types.ReplyKeyboardRemove())
        print(f"Получена фамилия {surname}")
    else:
        user_data[chat_id]['surname'] = "-"
        print("Выбраны все пользователи")
        bot.send_message(chat_id, 
                        "Введите номера контестов через пробел:", 
                        reply_markup=types.ReplyKeyboardRemove())

@bot.message_handler(func=lambda message: 'surname' in user_data.get(message.chat.id, {}) 
             and user_data.get(message.chat.id, {}).get('surname') != "-" 
             and 'name' not in user_data.get(message.chat.id, {}))
def get_name(message):
    """Получаем имя пользователя"""
    chat_id = message.chat.id
    user_data[chat_id]['name'] = message.text
    bot.send_message(chat_id, "Введите номера контестов через пробел:")
    print(f"Получено имя {user_data[chat_id]['name']}")

@bot.message_handler(func=lambda message: ('name' in user_data.get(message.chat.id, {}) 
             or user_data.get(message.chat.id, {}).get('surname') == "-") 
             and 'ids' not in user_data.get(message.chat.id, {}))
def get_contest_ids(message):
    """Получаем номера контестов"""
    chat_id = message.chat.id
    try:
        user_data[chat_id]['ids'] = list(map(int, message.text.split()))
        bot.send_message(chat_id, "Данные получены. Запускаю парсер...")
        print(f"Получены номера контестов: {user_data[chat_id]['ids']}")
        parser_wrapper(chat_id)
    except ValueError:
        bot.send_message(chat_id, "Ошибка: введите числа через пробел")
        del user_data[chat_id]

# Остальной код функции parser() остается без изменений
def parser(CONTEST_N, IS_IN_TIME, LOGIN, PASSWORD):
    """
    Парсит посылки пользователя с Codeforces
    Args:
        CONTEST_N (int): Номер контеста
        IS_IN_TIME (bool): Флаг временного контеста
        LOGIN (str): Логин пользователя Codeforces
        PASSWORD (str): Пароль пользователя Codeforces
    """
    try:
        # Отправляем сообщение о начале работы
        print(f"🟢 Начата обработка пользователя {LOGIN} для контеста {CONTEST_N}")
        
        driver = webdriver.Chrome()
        driver.maximize_window()
        
        # Шаг 1: Авторизация
        print(f"🔑 Попытка авторизации для {LOGIN}")
        driver.get("https://sirius0625.contest.codeforces.com/enter")
        
        # Ввод логина и пароля
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "handleOrEmail"))
        ).send_keys(LOGIN)
        
        driver.find_element(By.NAME, "password").send_keys(PASSWORD)
        driver.find_element(By.XPATH, "//input[@type='submit']").click()
        time.sleep(3)
        print(f"✅ Успешная авторизация для {LOGIN} с паролем {PASSWORD}")

        # Шаг 2: Переход к контесту
        contests = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="body"]/div[2]/div/ul/li[2]/a'))
        )
        contests.click()
        print("Открыл страницу с контестами")
        # Получаем название контеста
        contest_name = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, f'/html/body/div[6]/div[3]/div[2]/div/div/div[6]/table/tbody/tr[{CONTEST_N + 1}]/td[1]'))
        ).text.split("\n")[0]
        
        print(f"📌 Выбран контест: {contest_name}")

        # Шаг 3: Вход в контест
        try:
            enter = driver.find_element(
                By.XPATH, 
                f'//*[@id="pageContent"]/div/div/div[6]/table/tbody/tr[{CONTEST_N + 1}]/td[1]/a[1]')
            enter.click()
            time.sleep(1)
        except Exception as e:
            print(f"❌ Ошибка входа в контест: {str(e)}")
            driver.quit()
            return
        print("Вошел в контест")
        # Шаг 4: Переход к посылкам
        my_subs = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="pageContent"]/div[1]/ul/li[4]/a'))
        )
        my_subs.click()
        print("Открыл вкладку \"Мои посылки\"")
        # Шаг 5: Сбор данных о посылках
        try:
            try:
                subs = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="pageContent"]/div[7]/div[6]/table/tbody'))
                )
            except:
                subs = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="pageContent"]/div[2]/div[6]/table/tbody'))
                )
        except Exception as e:
            print(f"❌ Не найдены посылки: {str(e)}")
            driver.quit()
            return
        
        data = []
        children = subs.find_elements(By.XPATH, './*')[1:]  # Пропускаем заголовок
        send_log(f"🔍 Найдено {len(children)} посылок для анализа")
        print(f"🔍 Найдено {len(children)} посылок для анализа")
        SAVE_DIR = "solutions"
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        submission_rows = subs.find_elements(By.XPATH, './/tr[position()>1]')
        
        for idx, row in enumerate(submission_rows, 1):
            try:
                cells = row.find_elements(By.XPATH, './/td')
                if len(cells) < 6:
                    continue
                    
                submission_data = [
                    cells[0].text.strip(),  # ID
                    cells[1].text.strip(),  # Время
                    cells[2].text.strip(),  # Автор
                    cells[3].text.strip(),  # Задача
                    cells[4].text.strip(),  # Язык
                    cells[5].text.upper().strip()  # Вердикт
                ]
                print(f"Рассматривается посылка со следующими данными: ID:{submission_data[0]}, TIME:{submission_data[1]}, AUTHOR:{submission_data[2]}, TASK:{submission_data[3]}, LANGUAGE:{submission_data[4]}, VER:{submission_data[5]}")
                # Открываем посылку
                idd = driver.find_element(By.LINK_TEXT, submission_data[0])
                idd.click()
                print("Открыл решение")
                time.sleep(3)
                
                try:
                    # Получаем код напрямую из элемента
                    btn = driver.find_element(By.XPATH, '//*[@id="program-source-text-copy"]')
                    btn.click()
                    print("Нажата кнопка скопировать")
                    code = pyperclip.paste()
                    print(code)
                    ver = my_ai.check(classifier, '\\n '.join(code.split('\n')))
                    print(f"{ver}")
                    # Сохраняем в файл
                    problem_name = submission_data[3].replace(' ', '_')
                    filename = os.path.join(SAVE_DIR, f"{problem_name}_{submission_data[0]}.txt")
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Задача: {submission_data[3]}\n")
                        f.write(f"Автор: {submission_data[2]}\n")
                        f.write(f"Время: {submission_data[1]}\n")
                        f.write(f"Язык: {submission_data[4]}\n")
                        f.write(f"Вердикт: {submission_data[5]}\n")
                        f.write(f"Вероятность списывания: {ver}\n\n")
                        f.write(code)
                    print("Записал в файл решение")
                    message = (
                        f"✅ Решение задачи: {submission_data[3]}\n"
                        f"👤 Автор: {submission_data[2]}\n"
                        f"🆔 ID посылки: {submission_data[0]}\n"
                        f"✅ Вердикт: {submission_data[5]}\n"
                        f"🧑‍Язык: {submission_data[4]}\n"
                        f"🆔 Вероятность списывания: {ver}\n"
                    )
                    if flag:
                        if "py" in submission_data[5]:
                            s = "y"
                            if s == "y":
                                with open(filename, "rb") as f:
                                    bot.send_document(TELEGRAM_CHAT_ID, f, caption=message)
                                print("Переслал код")
                            else:
                                print("Отклонил пересылку кода")
                                send_log("⚠️  Попытка переслать код отклонена!")
                    else:
                        s = "y"
                        if s == "y":
                            with open(filename, "rb") as f:
                                bot.send_document(TELEGRAM_CHAT_ID, f, caption=message)
                            print("Переслал код")
                        else:
                            print("Отклонил пересылку кода")
                            send_log(message)
                            send_log("⚠️  Попытка переслать код отклонена!")
                except Exception as e:
                    print(f"⚠️ Ошибка получения кода посылки {submission_data[0]}: {str(e)}")
                time.sleep(1)
                # Закрываем окно
                try:
                    driver.find_element(By.CSS_SELECTOR, ".close").click()
                    time.sleep(0.5)
                    print("Закрыл решение")
                except:
                    pass
            except Exception as e:
                print(f"⚠️ Ошибка обработки строки {idx}: {str(e)}")
                continue
        
    except Exception as e:
        print(f"🔥 Критическая ошибка: {str(e)}")
    finally:
        driver.quit()
        send_log(f"🔴 Завершена обработка пользователя {LOGIN} для контеста {CONTEST_N}")
        print(f"🔴 Завершена обработка пользователя {LOGIN} для контеста {CONTEST_N}")

if __name__ == "__main__":
    #with open("")
    print("Бот запущен...")
    bot.polling(none_stop=True)
