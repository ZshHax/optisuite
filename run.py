import sys
import os
import subprocess
import time
import webbrowser
from streamlit.web import cli as stcli


def resolve_path(path):
    # Эта функция помогает exe-шнику найти файлы внутри себя
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.getcwd()
    return os.path.join(base_path, path)


def open_kiosk_mode():
    # Ждем пару секунд, пока сервер запустится
    time.sleep(3)
    url = "http://localhost:8501"

    # Пытаемся открыть в режиме КИОСКА (Полноэкранный без рамок)
    # Сначала Edge (есть на всех Windows), потом Chrome
    try:
        subprocess.Popen(f'start msedge --kiosk "{url}" --edge-kiosk-type=fullscreen', shell=True)
    except:
        try:
            subprocess.Popen(f'start chrome --kiosk "{url}"', shell=True)
        except:
            # Если ничего нет, открываем обычный браузер
            webbrowser.open(url)


if __name__ == "__main__":
    # Находим путь к твоему app.py внутри exe
    app_path = resolve_path("app.py")

    # Настройки запуска Streamlit (скрываем лишнее)
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.headless=true",
        "--theme.base=light",
        "--server.port=8501"
    ]

    # Запускаем открыватель браузера в отдельном потоке
    import threading

    threading.Thread(target=open_kiosk_mode).start()

    # Запускаем сам сервер
    sys.exit(stcli.main())