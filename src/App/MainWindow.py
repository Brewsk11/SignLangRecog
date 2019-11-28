import tkinter as tk
import numpy as np
from PIL import ImageTk, Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class MainWindow(tk.Tk):

    def __init__(self, settings, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.settings = settings
        self.letter_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                              'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.init_layout()


    def init_layout(self):
        unit_size = self.settings['gui_settings']['unit_size']

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

        self.video_feed = tk.Label(video_feed_f)
        self.video_feed.pack(expand=True, fill='both')
        

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

        f = Figure(figsize=(2, 3), dpi=72)
        self.prediction_info_axe = f.add_subplot(111)
        self.setup_pred_axes()
        self.prediction_info = FigureCanvasTkAgg(f, master=prediction_info_f)
        self.prediction_info.get_tk_widget().pack(expand=True, fill='both')

        # 3rd row
        application_info_f = init_frame(parent=information_frame, height=units(2), width=units(28),
                                        row=3, column=0, columnspan=3)

        self.app_info = tk.Label(application_info_f)
        self.app_info['text'] = "Initializing"
        self.app_info.pack(expand=True, fill='both')

    def load_new_frame(self, frame):
        # TODO: Mateusz
        # Wczytanie klatki do VIDEO FEED
        # Prefferably na osobnym wątku

        if True:  # If success:
            self.on_new_frame(frame)

    def on_new_frame(self, video_frame):
        # TODO: Mateusz
        # Znajdz reke w klatce i wczytaj do HAND VIEW
        # Prefferably na osobnym wątku
        size = (self.video_feed.winfo_width(), self.video_feed.winfo_height())
        video_frame = video_frame.resize(size=size)

        img = ImageTk.PhotoImage(video_frame)
        self.video_feed.configure(image=img)
        self.video_feed.image = img

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

    def on_hand_normalized(self, hand_norm: np.ndarray):
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

    def on_letter_classified(self, pred_list: np.ndarray):
        pred_list = pred_list.flatten()
        classified_letter = self.letter_labels[pred_list.argmax()]

        # Drawing
        # call the clear method on your axes
        self.prediction_info_axe.clear()
        self.setup_pred_axes(pred_list)

        # call the draw method on your canvas
        self.prediction_info.draw()

    def set_status(self, module_status):
        status_str = ""

        if not module_status['detector']:
            status_str += "Detector initializing... "

        if not module_status['normalizer']:
            status_str += "Normalizer initializing... "

        if not module_status['classifier']:
            status_str += "Classifier initializing... "

        if module_status['system']:
            status_str = "System ready."
            self.app_info['fg'] = 'darkgreen'

        self.app_info['text'] = status_str

    def setup_pred_axes(self, *args):
        xlabels = [str(i*10) + '%' for i in range(0, 11, 2)]
        ylabels = self.letter_labels
        for i in range(1, 5):
            self.prediction_info_axe.axvline(x=i, ymin=0, ymax=1, color='whitesmoke', alpha=0.3)

        if len(args) > 0:
            pred_list = args[0]
            index = np.arange(len(self.letter_labels))
            display_list = pred_list * 5
            self.prediction_info_axe.barh(index, display_list)

        self.prediction_info_axe.set_xticks(np.arange(len(xlabels)))
        self.prediction_info_axe.set_xticklabels(xlabels)
        self.prediction_info_axe.set_yticks(np.arange(len(ylabels)))
        self.prediction_info_axe.set_yticklabels(ylabels, fontsize='8')


