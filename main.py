import time
import csv

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Настройка драйвера (Chrome)
driver = webdriver.Chrome()

# Переход на страницу входа
driver.get("https://informatics.msk.ru/login/")

# Поиск полей ввода и ввод данных
username_field = driver.find_element(By.NAME, "username")
password_field = driver.find_element(By.NAME, "password")

username_field.send_keys("Daniil7212")
password_field.send_keys("DaN12OlimP72-17!")

# Нажатие кнопки входа
login_button = driver.find_element(By.XPATH, "//button[@type='submit']")

login_button.click()

button = driver.find_element(By.XPATH, '//*[@id="inst4931"]/div/div/ul/li[3]/div/div/div/a')
button.click()
m_max = driver.find_element(By.XPATH, '//*[@id="Pagination"]/ul/li[9]/a')
m_int = int(m_max.text) - 1

print(m_int)

data = []
for i in range(m_int):

    for j in range(1, 10):
        j *= 2
        data_h = []
        tr = driver.find_element(By.XPATH, f'/html/body/div[2]/div[3]/div/div/section/div/div[1]/div[19]/table/tbody/tr[{j}]')
        children = tr.find_elements(By.XPATH, './*')
        for k in children:
            data_h.append(k.text)

        button = tr.find_element(By.XPATH, f"//button[@class='btn btn-link' and @data_run_id='{data_h[0]}']")
        button.click()
        time.sleep(1)

        del data_h[5]
        if data_h[6] == "100":
            data_h[7] = "Accepted"
        else:
            data_h[7] = "Rejected"
        data.append(data_h)

        code = driver.find_element(By.XPATH, f'//*[@id="sourceTab{data_h[0]}"]/pre/code')
        time.sleep(1)
        button.click()


    reload = driver.find_element(By.XPATH, f"//button[contains(text(), 'Обновить')]")
    reload.click()
    time.sleep(1)
    nxt = driver.find_element(By.CSS_SELECTOR, f"a.page-link[data-page_id='{i + 1}']").find_element(By.XPATH, "..")
    time.sleep(0.5)
    print(nxt.text)
    nxt.click()
    time.sleep(0.25)


with open("output.csv", "w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerows(data)


# Закрытие браузера
driver.quit()
driver.get("https://informatics.msk.ru/login/")
