import tkinter as tk
from PIL import ImageTk
from App.Modules.DetectorAdapter import DetectorAdapterMockup
import matplotlib.pyplot as plt

class MainWindow(tk.Tk):

    def __init__(self, message_queue, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.init_layout()

        self.detector = DetectorAdapterMockup(message_queue)


    def init_layout(self):
        unit_size = 29

        def units(x):
            return x * unit_size

        padding_x = units(0.25)
        padding_y = units(0.25)

        def init_frame(parent, height, width, row, column, rowspan=1, columnspan=1, bg=None):
            frame = tk.Frame(parent, height=height, width=width, bg=bg)
            frame.grid(row=row, column=column,
                       rowspan=rowspan, columnspan=columnspan,
                       padx=int(padding_x), pady=int(padding_y), sticky='nesw')
            frame.pack_propagate(False)
            return frame

        information_frame = init_frame(parent=self, height=units(padding_y * 2 + 20), width=units(padding_x * 2 + 28),
                                       row=0, column=0, bg='lightgray')

        # 1st row
        predicted_text_f = init_frame(parent=information_frame, height=units(2), width=units(28),
                                      row=0, column=0, columnspan=3)

        predicted_text = tk.Label(predicted_text_f, text='')
        predicted_text.pack(expand=True, fill='both')
        predicted_text['font'] = ('Ubuntu Mono', '32')

        # 2nd row
        video_feed_f = init_frame(parent=information_frame, height=units(9), width=units(16),
                                       row=1, column=0, rowspan=2)

        hand_view_f = init_frame(parent=information_frame, height=units(4), width=units(4),
                                      row=1, column=1)

        self.image_label = tk.Label(hand_view_f)
        self.image_label['background'] = 'red'
        self.image_label.pack()

        normalized_view_f = init_frame(parent=information_frame, height=units(4), width=units(4),
                                            row=2, column=1)

        prediction_info_f = init_frame(parent=information_frame, height=units(9), width=units(6),
                                            row=1, column=2, rowspan=2)

        # 3rd row
        application_info_f = init_frame(parent=information_frame, height=units(2), width=units(28),
                                             row=3, column=0, columnspan=3)

    def load_new_frame(self, frame):
        # TODO: Mateusz
        # Wczytanie klatki do VIDEO FEED
        # Prefferably na osobnym wątku

        if True:  # If success:
            self.on_new_frame(frame)

    def on_new_frame(self, frame):
        # TODO: Mateusz
        # Znajdz reke w klatce i wczytaj do HAND VIEW
        # Prefferably na osobnym wątku

        if True:  # If hand_detected:
            pass
            # self.on_hand_detected(hand_img)

    def on_hand_detected(self, hand_img):
        # TODO: Pawel
        # Wczytaj zdjecie reki z HAND VIEW, wykonaj predict() na modelu Normalizatora i wyślij do NORMALIZED VIEW
        # Prefferably/must-have na osobnym wątku (operacje z siecią)

        img = ImageTk.PhotoImage(hand_img)
        self.image_label.configure(image=img)
        self.image_label.image = img

        print("on_hand_detected called")
        if True:  # If success:
            # self.on_hand_normalized(hand_norm)
            pass

    def on_hand_normalized(self, hand_norm):
        # TODO: Kuba
        # Wczytaj kontur z NORMALIZED VIEW, wykonaj predict() na modelu Klasyfikatora
        # Prefferably/must-have na osobnym wątku (operacje z siecią)

        if True:  # If success:
            # self.on_letter_classified(pred_list)
            pass

    def on_letter_classified(self, pred_list):
        # TODO: Kuba
        # Zczytaj prawodopodobienstwa predykcji, sprawdz czy ta sama litera nie była w poprzednich klatkach
        # Wyświetl listę prawdopodobieństw do PREDICTION VIEW

        pass
