import tkinter as tk
import numpy as np
from PIL import ImageTk, Image
from App.Modules.DetectorAdapter import DetectorAdapterMockup
from App.Modules.NormalizerAdapter import NormalizerAdapter
import matplotlib.pyplot as plt

class MainWindow(tk.Tk):

    def __init__(self, message_queue, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.init_layout()

        self.detector = DetectorAdapterMockup(message_queue)
        self.normalizer = NormalizerAdapter(message_queue)

    def init_layout(self):
        unit_size = 48

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

        self.hand_view = tk.Label(hand_view_f)
        self.hand_view['background'] = 'red'
        self.hand_view.pack(expand=True, fill='both')

        normalized_view_f = init_frame(parent=information_frame, height=units(4), width=units(4),
                                            row=2, column=1)

        self.normalized_view = tk.Label(normalized_view_f)
        self.normalized_view['background'] = 'blue'
        self.normalized_view.pack(expand=True, fill='both')

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

    def on_hand_detected(self, hand_img: Image.Image):
        # TODO: Pawel
        # Wczytaj zdjecie reki z HAND VIEW, wykonaj predict() na modelu Normalizatora i wyślij do NORMALIZED VIEW
        # Prefferably/must-have na osobnym wątku (operacje z siecią)

        # Resize the incoming image to expand inside the label
        size = (self.hand_view.winfo_width(), self.hand_view.winfo_height())
        hand_img = hand_img.resize(size=size)

        img = ImageTk.PhotoImage(hand_img)
        self.hand_view.configure(image=img)
        self.hand_view.image = img

        print("on_hand_detected called")

        self.normalizer.normalize(hand_img)

    def on_hand_normalized(self, hand_norm: np.array):
        # TODO: Kuba
        # Wczytaj kontur z NORMALIZED VIEW, wykonaj predict() na modelu Klasyfikatora
        # Prefferably/must-have na osobnym wątku (operacje z siecią)

        # Rescale the array to 0-255
        hand_norm *= 255
        hand_norm = np.uint32(hand_norm)

        pil_img = Image.fromarray(hand_norm)

        # Resize the  image to expand inside the label
        size = (self.normalized_view.winfo_width(), self.normalized_view.winfo_height())
        pil_img = pil_img.resize(size=size)

        img = ImageTk.PhotoImage(pil_img)
        self.normalized_view.configure(image=img)
        self.normalized_view.image = img

        if True:  # If success:
            # self.on_letter_classified(pred_list)
            pass

    def on_letter_classified(self, pred_list):
        # TODO: Kuba
        # Zczytaj prawodopodobienstwa predykcji, sprawdz czy ta sama litera nie była w poprzednich klatkach
        # Wyświetl listę prawdopodobieństw do PREDICTION VIEW

        pass
