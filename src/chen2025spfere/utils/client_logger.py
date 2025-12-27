import os, datetime

def fl_logger(message, filename=None):
    if filename is None:
        filename = f"client_log/log.log"

    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{timestamp}] {message}\n"

    with open(filename, "a") as file:
        file.write(full_message)

    print(full_message, end="")