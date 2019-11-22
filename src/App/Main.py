from App.MainWindow import MainWindow
from multiprocessing import Queue
from time import sleep

if __name__ == "__main__":

    task_queue = Queue()
    app = MainWindow(task_queue)

    while True:
        app.update_idletasks()
        app.update()

        while not task_queue.empty():
            message, payload = task_queue.get(block=False)
            print("Oh, got a message!: " + message)

            if message == "hand_detected":
                app.on_hand_detected(payload)

            elif message == "hand_normalized":
                app.on_hand_normalized(payload)

            elif message == "sign_classified":
                app.on_letter_classified(payload)
