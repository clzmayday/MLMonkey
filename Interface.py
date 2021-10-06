import tkinter as tk
from tkinter.font import Font
import time


def clear():
    for widge in app.winfo_children():
        widge.destroy()


def exit():
    app.destroy()


def begin():
    def load_dot():
        if time.time() - current_time > 5:
            loading["text"] = "Press any key to continue..."
        if loading["text"].endswith("......"):
            loading["text"] = "Loading"
        else:
            loading["text"] += "."

        if loading["text"].startswith("Loading"):
            app.after(250, load_dot)


    current_time = time.time()
    label = tk.Label(text="Machine Learning Monkey\nWindtex Estimation Regressor", font=("newspaper", 45, "bold"))
    label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
    loading = tk.Label(text="Loading", font=("newspaper", 30, "bold"))
    loading.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
    button = tk.Button(text="Quit", activeforeground="BLACK", height=2, justify=tk.CENTER, font=("newspaper", 25, "bold"),
                       command=exit)
    button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)
    app.after(1, load_dot)

def show():
    global page

    if page == 0:
        begin()


app = tk.Tk()
app.title("Machine Learning Monkey - Windtex Estimation Regressor")
app.geometry('1200x800')
page = 0
show()

app.mainloop()