import tkinter as tk
from tkinter.font import Font
import time


def clear():
    for widge in app.winfo_children():
        widge.destroy()

def clear_key():
    app.unbind("<Key>")
    app.unbind("<Escape>")
    app.unbind("<Return>")
    app.unbind("<Back>")
    app.unbind("<Button>")


def close():
    app.destroy()


def close_(event):
    app.destroy()


def after_loading(event):
    global page
    page = 1
    show()

def turn_page(p):
    global page
    page = p
    show()

def begin():
    def load_dot():
        if time.time() - current_time > 1.5:
            app.bind("<Key>", after_loading)
            app.bind("<Button>", after_loading)
            app.bind("<Escape>", close_)
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

    app.after(1, load_dot)


def inst_info():
    pass

def build_model():
    pass

def start():
    clear_key()
    label1 = tk.Label(text="Machine Learning Monkey Windtex Estimation Regressor", font=("newspaper", 25, "bold"))
    label1.place(relx=0.5, rely=0.05, anchor=tk.CENTER)
    label2 = tk.Label(text="Please click the button or press the key to continue", font=("newspaper", 15, "bold"))
    label2.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
    label3 = tk.Label(text="Step 1: Preparing the model", font=("newspaper", 20, "bold"), fg="RED")
    label3.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
    model_choice = tk.IntVar()
    check1 = tk.Radiobutton(text="Decision Tree", value=1, var=model_choice)
    check1.place(relx=0.3, rely=0.25, anchor=tk.CENTER)
    check2 = tk.Radiobutton(text="KNN", value=2, var=model_choice)
    check2.place(relx=0.4, rely=0.25, anchor=tk.CENTER)
    button1 = tk.Button(text="Build Model", activeforeground="RED", width=20, anchor="center",
                        font=("newspaper", 15, "bold"), command=lambda: turn_page(3))
    button1.place(relx=0.5, rely=0.3, anchor=tk.CENTER)


def menu():
    clear_key()
    label1 = tk.Label(text="Machine Learning Monkey Windtex Estimation Regressor", font=("newspaper", 30, "bold"))
    label1.place(relx=0.5, rely=0.25, anchor=tk.CENTER)
    label2 = tk.Label(text="Please click the button or press the key to continue", font=("newspaper", 20, "bold"))
    label2.place(relx=0.5, rely=0.35, anchor=tk.CENTER)
    button1 = tk.Button(text="1. Instruction and Information", activeforeground="RED", width=35, anchor="w",
                       font=("newspaper", 25, "bold"), command=close)
    button1.place(relx=0.5, rely=0.45, anchor=tk.CENTER)
    button2 = tk.Button(text="2. Windtex Estimation Regressor", activeforeground="RED", width=35, anchor="w",
                        font=("newspaper", 25, "bold"), command=lambda : turn_page(3))
    button2.place(relx=0.5, rely=0.55, anchor=tk.CENTER)
    button0 = tk.Button(text="0. Quit", activeforeground="RED", width=35, anchor="w",
                        font=("newspaper", 25, "bold"), command=close)
    button0.place(relx=0.5, rely=0.65, anchor=tk.CENTER)


def show():
    global page
    clear()
    if page == 0:
        begin()
    elif page == 1:
        menu()
    elif page == 3:
        start()



app = tk.Tk()
app.title("Machine Learning Monkey - Windtex Estimation Regressor")
app.geometry('1200x800')
page = 0
show()

app.mainloop()