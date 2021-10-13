import os, sys
import tkinter as tk
from tkinter.font import Font
import time
from MLMonkey import FeatureExtraction, Windtex, WindtexModel


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
    global status_message

    def load_dot():
        if time.time() - current_time > 1.5:
            app.bind("<Key>", after_loading)
            app.bind("<Button>", after_loading)
            app.bind("<Escape>", close_)

            loading["text"] = "Press any key to continue..."
        progress = round((time.time() - current_time) / 1.5 * 100)
        if progress > 100:
            progress = "100"
        else:
            progress = str(progress)
        status_message.set(
            "Status: Software starting......." + progress + "%")
        if loading["text"].endswith("......"):
            loading["text"] = "Loading"
        else:
            loading["text"] += "."

        if loading["text"].startswith("Loading"):
            app.after(250, load_dot)

    status_message.set("Status: Software Starting")
    current_time = time.time()
    label = tk.Label(text="Machine Learning Monkey\nWindtex Estimation Regressor", font=("newspaper", 45, "bold"))
    label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
    loading = tk.Label(text="Loading", font=("newspaper", 30, "bold"))
    loading.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    app.after(1, load_dot)


def inst_info():
    pass


def build_model(model_number):
    global status_message
    status_message.set("Feature Extracting...\nPlease wait...")
    app.update()
    data = Windtex.prepare("./Calculator.csv", "./data", "./label_2.json")
    status_message.set("Feature Extraction Done!\n"
                       "Please continue to the next step")
    app.update()


def update_status(button):
    global status_message, page
    if button == 1:
        status_message.set("Click to know how to use this software to use Machine Learning Monkey")
    elif button == -1:
        status_message.set("Click to close this software")
    elif button == 2:
        status_message.set("Click to train a Machine Learning Monkey and play with him")
    else:
        if page == 1:
            status_message.set("Status: Menu")
        elif page == 2:
            status_message.set("Status: ML Monkey Setting")


def start():
    global status_message
    clear_key()
    status_message.set("Status: ML Monkey Setting")
    label1 = tk.Label(text="Machine Learning Monkey Windtex Estimation Regressor", font=("newspaper", 25, "bold"))
    label1.place(relx=0.5, rely=0.05, anchor=tk.CENTER)
    label2 = tk.Label(text="Please click the button or press the key to continue", font=("newspaper", 15, "bold"))
    label2.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
    label3 = tk.Label(text="Step 1: Preparing the model", font=("newspaper", 20, "bold"), fg="RED")
    label3.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
    model_choice = tk.IntVar()
    model_choice.set(-1)
    check1 = tk.Radiobutton(text="Decision Tree Regression", value=1, var=model_choice)
    check1.place(relx=0.2, rely=0.25, anchor=tk.CENTER)
    check2 = tk.Radiobutton(text="K-Nearest Neighbour Regression", value=2, var=model_choice)
    check2.place(relx=0.4, rely=0.25, anchor=tk.CENTER)
    check3 = tk.Radiobutton(text="Logistic Regression", value=3, var=model_choice)
    check3.place(relx=0.6, rely=0.25, anchor=tk.CENTER)
    check4 = tk.Radiobutton(text="Linear Regression", value=4, var=model_choice)
    check4.place(relx=0.8, rely=0.25, anchor=tk.CENTER)
    check5 = tk.Radiobutton(text="Gaussian Process Regression", value=5, var=model_choice)
    check5.place(relx=0.2, rely=0.3, anchor=tk.CENTER)
    check6 = tk.Radiobutton(text="Bayesian Regression", value=6, var=model_choice)
    check6.place(relx=0.4, rely=0.3, anchor=tk.CENTER)
    check6 = tk.Radiobutton(text="Windtex Estimation Regression", value=-1, var=model_choice)
    check6.place(relx=0.8, rely=0.3, anchor=tk.CENTER)
    button1 = tk.Button(text="Build Model", activeforeground="RED", width=20, anchor="center",
                        font=("newspaper", 15, "bold"), command=lambda: build_model(model_choice.get()))
    button1.place(relx=0.5, rely=0.35, anchor=tk.CENTER)


def menu():
    global status_message
    clear_key()
    status_message.set("Status: Menu")
    label1 = tk.Label(text="Machine Learning Monkey Windtex Estimation Regressor", font=("newspaper", 30, "bold"))
    label1.place(relx=0.5, rely=0.25, anchor=tk.CENTER)
    label2 = tk.Label(text="Please click the button or press the key to continue", font=("newspaper", 20, "bold"))
    label2.place(relx=0.5, rely=0.35, anchor=tk.CENTER)
    button1 = tk.Button(text="1. Instruction and Information", activeforeground="RED", width=35, anchor="w",
                        font=("newspaper", 25, "bold"), command=close)
    button1.place(relx=0.5, rely=0.45, anchor=tk.CENTER)
    button1.bind("<Enter>", lambda x: update_status(1))
    button1.bind("<Leave>", lambda x: update_status(0))
    button2 = tk.Button(text="2. Windtex Estimation Regressor", activeforeground="RED", width=35, anchor="w",
                        font=("newspaper", 25, "bold"), command=lambda: turn_page(3))
    button2.place(relx=0.5, rely=0.55, anchor=tk.CENTER)
    button2.bind("<Enter>", lambda x: update_status(2))
    button2.bind("<Leave>", lambda x: update_status(0))
    button0 = tk.Button(text="0. Quit", activeforeground="RED", width=35, anchor="w",
                        font=("newspaper", 25, "bold"), command=close)
    button0.place(relx=0.5, rely=0.65, anchor=tk.CENTER)
    button0.bind("<Enter>", lambda x: update_status(-1))
    button0.bind("<Leave>", lambda x: update_status(0))


def show():
    global page, status_message
    clear()
    label0 = tk.Label(text="", bg="WHITE", justify="left", width=150, height=3, textvariable=status_message,
                      font=("newspaper", 15, "bold"))
    label0.place(relx=0.5, rely=1, anchor="s")
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
status_message = tk.StringVar(app, "Status: Ready")
show()
app.mainloop()
