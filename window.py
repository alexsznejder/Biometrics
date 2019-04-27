import math
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from PIL import ImageTk, Image
import imageio
import matplotlib.pyplot as plt


class Picture():
    def __init__(self):
        self.root = Tk()
        self.root.geometry("240x280")
        min_string = StringVar()
        max_string = StringVar()

        button_add = Button(self.root, text="Add picture", command=self.add_pic)
        button_add.pack()
        button_add.place(x=10, y=10, height=40, width=100)
        button_save = Button(self.root, text="Save", command=self.save)
        button_save.pack()
        button_save.place(x=130, y=10, height=40, width=100)

        button_bright = Button(self.root, text="Bright", command=self.bright)
        button_bright.pack()
        button_bright.place(x=10, y=70, height=40, width=100)
        button_dark = Button(self.root, text="Dark", command=self.dark)
        button_dark.pack()
        button_dark.place(x=130, y=70, height=40, width=100)

        Label(self.root, text="min").place(x=10, y=130, height=20, width=30)
        self.min_entry = Entry(self.root, textvariable=min_string)
        self.min_entry.pack()
        self.min_entry.place(x=45, y=130, height=20, width=60)
        Label(self.root, text="max").place(x=130, y=130, height=20, width=30)
        self.max_entry = Entry(self.root, textvariable=max_string)
        self.max_entry.pack()
        self.max_entry.place(x=165, y=130, height=20, width=60)

        button_alig = Button(self.root, text="Alignment", command=self.alignment, height=2, width=12)
        button_alig.pack()
        button_alig.place(x=10, y=170, height=40, width=100)
        button_stretch = Button(self.root, text="Stretching", command=self.stretching)
        button_stretch.pack()
        button_stretch.place(x=130, y=170, height=40, width=100)

        button_hist = Button(self.root, text="Histograms", command=self.hieroglify)
        button_hist.pack()
        button_hist.place(x=70, y=230, height=40, width=100)

        self.root.mainloop()

    def add_pic(self):
        self.file_path = filedialog.askopenfilename()
        if not self.file_path[-3:] == 'gif':
            self.imageCV = cv2.cvtColor(cv2.imread(self.file_path), cv2.COLOR_BGR2RGB)
        else:
            gif = imageio.mimread(self.file_path)
            self.imageCV = [img for img in gif][0]

        oryg = Toplevel(self.root) #wyświetlanie w nowym oknie
        oryg.title("Oryginalny")
        label = Label(oryg)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV))
        label.configure(image=self.photo)
        label.image = self.photo
        label.pack()

        height, width, nie_potrzebne = self.imageCV.shape
        r_string = StringVar()
        g_string = StringVar()
        b_string = StringVar()
        self.rgb = np.zeros((height, width))
        for x in range(len(self.imageCV)):
            for y in range(len(self.imageCV[x])):
                self.rgb[x, y] = int(int(self.imageCV[x, y, 0]) + int(self.imageCV[x, y, 1]) + int(self.imageCV[x, y, 2]))/3

        self.roboczy = Toplevel(self.root)
        self.roboczy.title("Zmieniany")
        self.roboczy.minsize(width+30, height+150)
        self.canvas = Canvas(self.roboczy, width=width, height=height)
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.canvas.bind("<Motion>", self.cut)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Leave>", self.to_normal)
        self.canvas.bind("<Button-1>", self.change_pixel)
        self.canvas.pack()
        self.canvas.place(x=15, y=10)
        frame = Frame(self.roboczy)
        frame.pack(side=BOTTOM, fill=Y, pady=15)

        self.label_pixel = Label(self.roboczy)
        self.label_pixel.pack(side=BOTTOM, pady=10)
        self.label_zoom = Label(self.roboczy)
        self.label_zoom.pack(side=BOTTOM)

        Label(frame, text="R").grid(row=0, column=0, pady=5, sticky=SW) #do wprowadzania RGB
        self.r_entry = Entry(frame, textvariable=r_string, width=3)
        self.r_entry.grid(row=0, column=1, pady=5, padx=5, sticky=SW)

        Label(frame, text="G").grid(row=0, column=2, pady=5, sticky=SW)
        self.g_entry = Entry(frame, textvariable=g_string, width=3)
        self.g_entry.grid(row=0, column=3, padx=5, pady=5)

        Label(frame, text="B").grid(row=0, column=4, pady=5, sticky=SW)
        self.b_entry = Entry(frame, textvariable=b_string, width=3)
        self.b_entry.grid(row=0, column=5, padx=5, pady=5)

        menubar = Menu(self.roboczy)
        algmenu = Menu(menubar, tearoff=0)
        algmenu.add_command(label="Gray", command=self.to_gray)
        algmenu.add_command(label="Binarization", command=self.binar_window)
        algmenu.add_command(label="Otsu", command=self.otsu)
        algmenu.add_command(label="Niblack", command=self.niblack_window)
        menubar.add_cascade(label="Binarization", menu=algmenu)

        filmenu = Menu(menubar, tearoff=0)
        filmenu.add_command(label="Własny", command=self.wlasny_window)
        filmenu.add_command(label="Box filter", command=self.box_filter)
        filmenu.add_command(label="Prewitt", command=self.prewitt)
        filmenu.add_command(label="Sobel", command=self.sobel)
        filmenu.add_command(label="Laplace", command=self.laplace)
        filmenu.add_command(label="Corners", command=self.corners)
        filmenu.add_command(label="Kuwahara", command=self.kuwahara)
        filmenu.add_command(label="Medianowy 3x3", command=lambda: self.median(3))
        filmenu.add_command(label="Medianowy 5x5", command=lambda: self.median(5))
        menubar.add_cascade(label="Filters", menu=filmenu)
        self.roboczy.config(menu=menubar)

        self.zoomcycle = 0
        self.zimg_id = None

    def wlasny_window(self):
        self.m_string = StringVar()
        okienko = Toplevel(self.roboczy)
        okienko.maxsize(215, 140)

        Label(okienko, text="m :").place(x=40, y=40, height=20, width=20)
        self.m_entry = Entry(okienko, textvariable=self.m_string)
        self.m_entry.pack()
        self.m_entry.place(x=65, y=40, height=20, width=105)

        button_entry = Button(okienko, text="Entry", command=self.wlasny)
        button_entry.pack()
        button_entry.place(x=55, y=80, height=40, width=100)

    def wlasny(self):
        matrix = np.fromstring(self.m_string.get(), dtype=int, sep=", ")
        print(matrix)
        self.filter(matrix)

    def box_filter(self):
        matrix = np.full(9, 1)
        self.filter(matrix)

    def filter(self, matrix):
        new_image = np.zeros(self.imageCV.shape)
        for ch in range(3):
            for x in range(len(self.imageCV)):
                for y in range(len(self.imageCV[x])):
                    sum, n = 0, 0
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            try:
                                if x + i >= 0 and y + j >= 0:
                                    sum += matrix[n] * self.imageCV[x + i, y + j, ch]
                                    n += 1
                                else:
                                    sum += 0
                            except IndexError:
                                sum += 0
                    if sum > 255:
                        sum = 255
                    elif sum < 0:
                        sum = 0
                    #sum = sum/n
                    new_image[x, y, ch] = int(sum)

        self.imageCV = new_image
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def prewitt(self):
        #matrix1 = [-1, -1, 0, -1, 0, 1, 0, 1, 1]
        matrix1 = [0, 1, 1, -1, 0, 1, -1, -1, 0]
        self.filter(matrix1)

    def sobel(self):
        matrix = [2, 1, 0, 1, 0, -1, 0, -1, -2]
        self.filter(matrix)

    def laplace(self):
        matrix1 = [0, -1, 0, -1, 4, -1, 0, -1, 0]
        self.filter(matrix1)

    def corners(self):
        matrix = [1, -1, -1, 1, -2, -1, 1, 1, 1]
        self.filter(matrix)

    def kuwahara(self):
        #np.var wariancja
        new_image = np.zeros(self.imageCV.shape)
        n = -2
        m = -2
        for ch in range(3):
            for x in range(len(self.imageCV)):
                for y in range(len(self.imageCV[x])):
                    for box in range(4):
                        pixels = []
                        variations = []
                        mean = []
                        for i in range(n, n+3):
                            for j in range(m, m+3):
                                try:
                                    if x + i >= 0 and y + j >= 0:
                                        pixels.append(self.imageCV[x + i, y + j, ch])
                                except IndexError:
                                    pass
                        if box == 1:
                            m = 0
                        if box == 2:
                            n = 0
                        if box == 3:
                            m = -2
                        mean.append(np.mean(pixels))
                        variations.append(np.var(pixels))
                    min_var = min(variations)
                    new_image[x, y, ch] = int(mean[variations.index(min_var)])

        self.imageCV = new_image
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def median(self, size):
        size = int(size/2)
        print(size)
        new_image = np.zeros(self.imageCV.shape)
        for ch in range(3):
            for x in range(len(self.imageCV)):
                for y in range(len(self.imageCV[x])):
                    pixels = []
                    for i in range(-size, size+1):
                        for j in range(-size, size+1):
                            try:
                                if x + i >= 0 and y + j >= 0:
                                    pixels.append(self.imageCV[x + i, y + j, ch])
                            except IndexError:
                                pass
                    new_image[x, y, ch] = int(np.median(pixels))

        self.imageCV = new_image
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def bright(self):
        lighter = np.vectorize(lambda x: min(int(30 * math.log2(x + 1)), 255))
        self.imageCV = lighter(self.imageCV)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def dark(self):
        darker = np.vectorize(lambda x: min(int(0.003*(math.pow(x, 2))), 255))
        self.imageCV = darker(self.imageCV)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def stretching(self):
        k_min = int(self.min_entry.get())
        k_max = int(self.max_entry.get())
        for i in range(3):
            for x in range(len(self.imageCV)):
                for y in range(len(self.imageCV[x])):
                    if self.imageCV[x, y, i] >= k_min and self.imageCV[x, y, i] <= k_max:
                        self.imageCV[x, y, i] = int(((self.imageCV[x, y, i] - k_min)/(k_max - k_min))*255)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def to_gray(self):
        for ch in range(3):
            for x in range(len(self.imageCV)):
                for y in range(len(self.imageCV[x])):
                    self.imageCV[x, y, ch] = self.rgb[x, y]

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def binar_window(self):
        t_string = StringVar()
        okienko = Toplevel(self.roboczy)
        okienko.maxsize(170, 190)

        Label(okienko, text="t :").place(x=50, y=50, height=20, width=20)
        self.t_entry = Entry(okienko, textvariable=t_string)
        self.t_entry.pack()
        self.t_entry.place(x=75, y=50, height=20, width=40)

        button_entry = Button(okienko, text="Entry", command=self.binarization)
        button_entry.pack()
        button_entry.place(x=35, y=100, height=40, width=100)

    def binarization(self):
        t = int(self.t_entry.get())
        for ch in range(3):
            for x in range(len(self.imageCV)):
                for y in range(len(self.imageCV[x])):
                    if self.imageCV[x, y, ch] <= t:
                        self.imageCV[x, y, ch] = 0
                    else:
                        self.imageCV[x, y, ch] = 255

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def otsu(self):
        rgb_unique, rgb_counts = np.unique(self.rgb, return_counts=True)
        rgb = dict(zip(rgb_unique, rgb_counts))
        height, width, _ = self.imageCV.shape
        size = height * width
        ow_min = np.inf

        for i in range(255):
            p1, p2, u1, u2, o1, o2 = 0, 0, 0, 0, 0, 0
            for key, value in rgb.items():
                if key <= i:
                    p1 += value/size
                else:
                    p2 += value/size
            for key, value in rgb.items():
                if key <= i:
                    u1 += key * (value / size) / p1
                else:
                    u2 += key * (value / size) / p2
            for key, value in rgb.items():
                if key <= i:
                    o1 += ((key - u1) ** 2) * ((value / size) / p1)
                else:
                    o2 += ((key - u2) ** 2) * ((value / size) / p2)
            ow = (p1 * o1) + (p2 * o2)
            if ow < ow_min:
                ow_min = ow
                t = i

        for ch in range(3):
            for x in range(len(self.imageCV)-1):
                for y in range(len(self.imageCV[x]-1)):
                    if self.imageCV[x, y, ch] <= t:
                        self.imageCV[x, y, ch] = 0
                    else:
                        self.imageCV[x, y, ch] = 255

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def niblack_window(self):
        n_string = StringVar()
        k_string = StringVar()
        okienko = Toplevel(self.roboczy)
        okienko.maxsize(170, 210)

        Label(okienko, text="n :").place(x=50, y=40, height=20, width=20)
        self.n_entry = Entry(okienko, textvariable=n_string)
        self.n_entry.pack()
        self.n_entry.place(x=75, y=40, height=20, width=40)

        Label(okienko, text="k :").place(x=50, y=80, height=20, width=20)
        self.k_entry = Entry(okienko, textvariable=k_string)
        self.k_entry.pack()
        self.k_entry.place(x=75, y=80, height=20, width=40)

        button_entry = Button(okienko, text="Entry", command=self.niblack)
        button_entry.pack()
        button_entry.place(x=35, y=120, height=40, width=100)

    def niblack(self):
        new_image = np.zeros(self.imageCV.shape)
        k = float(self.k_entry.get())
        n = int(self.n_entry.get())
        n = int(n/2)
        for x in range(len(self.imageCV)):
            for y in range(len(self.imageCV[x])):
                pixels = []
                for i in range(-n, n+1):
                    for j in range(-n, n+1):
                        try:
                            if x + i >= 0 and y + j >= 0:
                                if not (i == 0 and j == 0):
                                    pixels.append(self.imageCV[x + i, y + j, 0])
                            else:
                                pixels.append(0)
                        except IndexError:
                            pixels.append(0)
                t = np.mean(pixels) + (k * np.std(pixels))
                for ch in range(3):
                    if self.imageCV[x, y, ch] <= t:
                        new_image[x, y, ch] = 255
                    else:
                        new_image[x, y, ch] = 0

        self.imageCV = new_image
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def hieroglify(self):
        r = self.imageCV[:, :, 0]
        r_unique, r_counts = np.unique(r, return_counts=True)
        r = dict(zip(r_unique, r_counts))

        g = self.imageCV[:, :, 1]
        g_unique, g_counts = np.unique(g, return_counts=True)
        g = dict(zip(g_unique, g_counts))

        b = self.imageCV[:, :, 2]
        b_unique, b_counts = np.unique(b, return_counts=True)
        b = dict(zip(b_unique, b_counts))

        rgb_unique, rgb_counts = np.unique(self.rgb, return_counts=True)
        rgb = dict(zip(rgb_unique, rgb_counts))

        plt.subplot(2, 2, 1)
        plt.bar(r.keys(), r.values(), color='r')
        plt.title("Red")

        plt.subplot(2, 2, 2)
        plt.bar(g.keys(), g.values(), color='g')
        plt.title("Green")

        plt.subplot(2, 2, 3)
        plt.bar(b.keys(), b.values(), color='b')
        plt.title("Blue")

        plt.subplot(2, 2, 4)
        plt.bar(rgb.keys(), rgb.values(), color='#838383')
        plt.title("RGB")
        plt.show()

    def lut(self, channel):
        r = self.imageCV[:, :, channel]
        r_unique, r_counts = np.unique(r, return_counts=True)
        r = dict(zip(r_unique, r_counts))

        height, width, _ = self.imageCV.shape
        size = height * width
        self.r_dys = np.zeros(256)
        for key, value in r.items():
            self.r_dys[key] = value/size
        self.r_dys = np.cumsum(self.r_dys)

        LUT = np.zeros(256)
        for i in range(len(self.r_dys)):
            LUT[i] = ((self.r_dys[i] - self.r_dys[0])/(1 - self.r_dys[0]))*255

        for x in range(width-1):
            for y in range(height-1):
                self.imageCV[x, y, channel] = int(LUT[self.imageCV[x, y, channel]])

    def alignment(self):
        for i in range(3):
            self.lut(i)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imageCV.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def to_normal(self, event):
        self.canvas.delete(self.zimg_id)

    def zoom(self, event):
        if (event.delta > 0):
            if self.zoomcycle != 4:
                self.zoomcycle += 1
        elif (event.delta < 0):
            if self.zoomcycle != 0:
                self.zoomcycle -= 1
        self.cut(event)

    def refresh(self):
        self.image = Image.fromarray(self.imageCV.astype('uint8'))
        new_ph = ImageTk.PhotoImage(image=self.image)
        return new_ph

    def cut(self, event):
        self.x = event.x
        self.y = event.y
        try:
            RGB = self.imageCV[event.y, event.x]
            self.label_pixel.configure(text="RGB" + str(RGB))
        except (IndexError, UnboundLocalError):
            print('wyszlam')

        if self.zimg_id:
            self.canvas.delete(self.zimg_id)
        self.label_zoom.configure(text="")
        if self.zoomcycle != 0:
            x, y = event.x, event.y
            if self.zoomcycle == 1:
                tmp = Image.fromarray(self.imageCV.astype('uint8')).crop((x - 45, y - 30, x + 45, y + 30))
            elif self.zoomcycle == 2:
                tmp = Image.fromarray(self.imageCV.astype('uint8')).crop((x - 30, y - 20, x + 30, y + 20))
            elif self.zoomcycle == 3:
                tmp = Image.fromarray(self.imageCV.astype('uint8')).crop((x - 15, y - 10, x + 15, y + 10))
            elif self.zoomcycle == 4:
                tmp = Image.fromarray(self.imageCV.astype('uint8')).crop((x - 6, y - 4, x + 6, y + 4))
            size = 300, 200
            self.label_zoom.configure(text="x" + str(2 ** self.zoomcycle))
            self.zimg = ImageTk.PhotoImage(tmp.resize(size))

            self.zimg_id = self.canvas.create_image(event.x, event.y, image=self.zimg)

    def change_pixel(self, event):
        try:
            self.imageCV[event.y, event.x] = [self.r_entry.get(), self.g_entry.get(), self.b_entry.get()]
        except ValueError:
            print('zla wartosc')
        self.new_ph = self.refresh()
        self.canvas.create_image(0, 0, image=self.new_ph, anchor=NW)
        self.canvas.pack()
        self.cut(event)

    def save(self):
        f = filedialog.asksaveasfile(mode="w", defaultextension=".png", filetypes=(("PNG file", "*.png"), ("All Files", "*.*"), ("JPG", "*.jpg"), ("GIF", "*.gif"), ("TIF", "*.tif"), ("BMP", "*.bmp")))
        if f:
            self.image.save(f.name)


if __name__ == "__main__":
    P = Picture()