import time
import csv
import pyperclip

from selenium import webdriver
from selenium.webdriver.common.by import By


def parser(CONTEST_N, IS_IN_TIME, LOGIN, PASSWORD):
    # Настройка драйвера (Chrome)
    driver = webdriver.Chrome()

    # Переход на страницу входа
    driver.get("https://sirius0625.contest.codeforces.com/enter")

    # Поиск полей ввода и ввод данных
    username_field = driver.find_element(By.NAME, "handleOrEmail")
    password_field = driver.find_element(By.NAME, "password")

    username_field.send_keys(LOGIN)
    password_field.send_keys(PASSWORD)

    # Нажатие кнопки входа
    login_button = driver.find_element(By.XPATH, "//input[@type='submit']")
    login_button.click()
    time.sleep(5)

    contests = driver.find_element(By.XPATH, '//*[@id="body"]/div[2]/div/ul/li[2]/a')
    contests.click()

    groups = driver.find_element(By.XPATH, '//*[@id="pageContent"]/div/div/div[6]/table/tbody')
    groups = groups.find_elements(By.XPATH, './*')

    enter = driver.find_element(By.XPATH, f'//*[@id="pageContent"]/div/div/div[6]/table/tbody/tr[{CONTEST_N + 1}]/td[1]/a[1]')
    enter.click()
    time.sleep(0.5)

    my_subs = driver.find_element(By.XPATH, f'//*[@id="pageContent"]/div[1]/ul/li[4]/a')
    my_subs.click()

    data = []
    if IS_IN_TIME:
        subs = driver.find_element(By.XPATH, '//*[@id="pageContent"]/div[7]/div[6]/table/tbody')
    else:
        subs = driver.find_element(By.XPATH, '//*[@id="pageContent"]/div[2]/div[6]/table/tbody')

    children = subs.find_elements(By.XPATH, './*')
    for k in children[1:]:
        data_h = []
        for p in k.find_elements(By.XPATH, './*'):
            data_h.append(p.text)
        data.append(data_h)

        idd = driver.find_element(By.LINK_TEXT, f"{data_h[0]}")
        idd.click()
        time.sleep(0.5)

        copy = driver.find_element(By.ID, "program-source-text-copy")
        copy.click()
        time.sleep(0.2)

        copied_text = pyperclip.paste()
        with open(f'codes/{data_h[0]}.txt', 'w', encoding='utf-8') as file:
            file.write(copied_text)
        time.sleep(0.2)

        idds = driver.find_elements(By.CLASS_NAME, "close")
        for i in idds:
            try:
                i.click()
            except:
                pass
        time.sleep(0.5)

    with open("output.csv", "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(data)

    driver.quit()
