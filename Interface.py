import os, sys, pickle
import tkinter as tk
from tkinter.font import Font
from tkinter import messagebox
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
    labelv = tk.Label(text="Version " + version, font=("newspaper", 15, "bold"))
    labelv.place(relx=0.9, rely=0.9, anchor=tk.CENTER)
    loading = tk.Label(text="Loading", font=("newspaper", 30, "bold"))
    loading.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    app.after(1, load_dot)


def inst_info():
    pass


def build_model():
    global status_message, page, model_choice, model
    data = None
    m_choice = model_choice.get()
    label1 = tk.Label(text="Machine Learning Monkey Windtex Estimation Regressor", font=("newspaper", 30, "bold"))
    label1.place(relx=0.5, rely=0.05, anchor=tk.CENTER)

    label2 = tk.Label(text="ML Monkey Model Setting Done!\nSelected Model: " + model_list[m_choice], font=("newspaper", 30, "bold"))
    label2.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label3 = tk.Label(text="Feature Extraction", font=("newspaper", 25, "bold"), fg="BLUE")
    label3.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    label4 = tk.Label(text="Model Initialisation", font=("newspaper", 25, "bold"), fg="BLACK")
    label4.place(relx=0.5, rely=0.45, anchor=tk.CENTER)
    label5 = tk.Label(text="Model Training", font=("newspaper", 25, "bold"), fg="BLACK")
    label5.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    label5 = tk.Label(text="Model Validation", font=("newspaper", 25, "bold"), fg="BLACK")
    label5.place(relx=0.5, rely=0.55, anchor=tk.CENTER)
    button3 = tk.Button(text="Main Menu", activeforeground="RED", width=25, anchor="center",
                        font=("newspaper", 20, "bold"), command=lambda: turn_page(1))
    button3.place(relx=0.7, rely=0.85, anchor=tk.CENTER)
    button2 = tk.Button(text="Back", activeforeground="RED", width=25, anchor="center",
                        font=("newspaper", 20, "bold"), command=lambda: turn_page(3))
    button2.place(relx=0.7, rely=0.85, anchor=tk.CENTER)
    if m_choice == -1:
        status_message.set("Feature Extracting... Please wait...")
        app.update()
        data = Windtex.prepare("./Calculator.csv", "./data", "./label_2.json")
        status_message.set("Feature Extraction Done!\n"
                           "Initialising Model... Please wait...")
        app.update()
        label3["fg"] = "BLACK"
        label4["fg"] = "BLUE"
        WindtexModel.born(data)
    if model is None:
        messagebox.showerror("No model loaded", "Error: No model loaded\nPlease upload again!")
        status_message.set("Uploaded model is not found! Please upload again!")
        app.update()
        turn_page(3)


def update_status(button):
    global status_message, page, model_choice
    if button == "b1":
        status_message.set("Click to know how to use this software to use Machine Learning Monkey")
    elif button == "quit":
        status_message.set("Click to close this software")
    elif button == "b2":
        status_message.set("Click to train a Machine Learning Monkey and play with him")
    elif button == "adaboost":
        status_message.set('Ada Boost Regression(base_estimator=Decision Tree(criterion="mse"), '
                           'learning_rate=0.00001, n_estimator=200)'
                           '\nMean Absolute Error: 0.71-0.74')
    elif button == "dt":
        status_message.set('Decision Tree Regression(criterion="mse")'
                           '\nMean Absolute Error: 1.01-1.06')
    elif button == "svr":
        status_message.set('Support Vector Regression(kernel="poly",degree=4,epsilon=0.1,coef0=4)'
                           '\nMean Absolute Error: 0.95-0.96')
    elif button == "huber":
        status_message.set('Huber Regression(max_interation=2000,epsilon=1.5)'
                           '\nMean Absolute Error: 0.98-0.99')
    elif button == "log":
        status_message.set('Logistic Regression(solver="liblinear", max_iteration=5000)'
                           '\nMean Absolute Error: 0.98-0.99')
    elif button == "linear":
        status_message.set('Linear Regression()'
                           '\nMean Absolute Error: 0.96')
    elif button == "kridge":
        status_message.set('Kernel Ridge(alpha=1,kernel="linear",degree=1)'
                           '\nMean Absolute Error: 0.95-0.96')
    elif button == "custom":
        status_message.set('Please store the algorithm model into a pickle file and upload here')
    else:
        if page == 1:
            status_message.set("Status: Menu")
        elif page == 3:
            status_message.set("Status: ML Monkey Setting\nSelected Model: " + model_list[model_choice.get()])
        elif page == 4:
            status_message.set("Status: ML Monkey Loading/Training")


def start():
    global status_message, model, model_choice

    def upload_model():
        global model
        from tkinter import filedialog
        model_path = filedialog.askopenfilename()
        with open(os.path.abspath(model_path), "rb") as pkl_file:
            model = pickle.load(pkl_file)
            pkl_file.close()
        if "fit" not in dir(model):
            messagebox.showerror("No fit Function", "Error: No fit() function inside of the model\n"
                                                       "Please upload again!")
            model = None
        if "predict" not in dir(model):
            messagebox.showerror("No predict Function", "Error: No predict() function inside of the model"
                                                           "\n Please upload again!")
            model = None
        if model is not None:
            button0["text"] = "Custom Model Upload - Done"
    clear_key()
    model_choice.set(1)
    status_message.set("Status: ML Monkey Setting\nSelected Model: " + model_list[model_choice.get()])
    label1 = tk.Label(text="Machine Learning Monkey Windtex Estimation Regressor", font=("newspaper", 30, "bold"))
    label1.place(relx=0.5, rely=0.05, anchor=tk.CENTER)
    label2 = tk.Label(text="Please click the button or press the key to continue", font=("newspaper", 20, "bold"))
    label2.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
    label3 = tk.Label(text="Step 1: Preparing the model", font=("newspaper", 25, "bold"), fg="RED")
    label3.place(relx=0.3, rely=0.2, anchor=tk.CENTER)
    label4 = tk.Label(text="Step 2: ML Monkey Options", font=("newspaper", 25, "bold"), fg="RED")
    label4.place(relx=0.8, rely=0.2, anchor=tk.CENTER)

    check1 = tk.Radiobutton(text="1. Ada Boost Regression - Decision Tree Regression", value=1, font=("newspaper", 20, "bold"), var=model_choice)
    check1.place(relx=0.3, rely=0.25, anchor=tk.CENTER)
    check1.bind("<Enter>", lambda x: update_status("adaboost"))
    check1.bind("<Leave>", lambda x: update_status(""))
    check2 = tk.Radiobutton(text="2. Decision Tree Regression", value=2, var=model_choice, font=("newspaper", 20, "bold"))
    check2.place(relx=0.3, rely=0.3, anchor=tk.CENTER)
    check2.bind("<Enter>", lambda x: update_status("dt"))
    check2.bind("<Leave>", lambda x: update_status(""))
    check3 = tk.Radiobutton(text="3. Support Vector Regression", value=3, var=model_choice, font=("newspaper", 20, "bold"))
    check3.place(relx=0.3, rely=0.35, anchor=tk.CENTER)
    check3.bind("<Enter>", lambda x: update_status("svr"))
    check3.bind("<Leave>", lambda x: update_status(""))
    check4 = tk.Radiobutton(text="4. Huber Regression", value=4, var=model_choice, font=("newspaper", 20, "bold"))
    check4.place(relx=0.3, rely=0.4, anchor=tk.CENTER)
    check4.bind("<Enter>", lambda x: update_status("huber"))
    check4.bind("<Leave>", lambda x: update_status(""))
    check5 = tk.Radiobutton(text="5. Logistic Regression", value=5, var=model_choice, font=("newspaper", 20, "bold"))
    check5.place(relx=0.3, rely=0.45, anchor=tk.CENTER)
    check5.bind("<Enter>", lambda x: update_status("log"))
    check5.bind("<Leave>", lambda x: update_status(""))
    check6 = tk.Radiobutton(text="6. Linear Regression", value=6, var=model_choice, font=("newspaper", 20, "bold"))
    check6.place(relx=0.3, rely=0.5, anchor=tk.CENTER)
    check6.bind("<Enter>", lambda x: update_status("linear"))
    check6.bind("<Leave>", lambda x: update_status(""))
    check7 = tk.Radiobutton(text="7. Kernel Ridge", value=7, var=model_choice, font=("newspaper", 20, "bold"))
    check7.place(relx=0.3, rely=0.55, anchor=tk.CENTER)
    check7.bind("<Enter>", lambda x: update_status("kridge"))
    check7.bind("<Leave>", lambda x: update_status(""))
    check0 = tk.Radiobutton(text="Custom Model (required fit() and predict() function)", value=-1, var=model_choice,
                            font=("newspaper", 20, "bold"))
    check0.place(relx=0.3, rely=0.6, anchor=tk.CENTER)
    check0.bind("<Enter>", lambda x: update_status("custom"))
    check0.bind("<Leave>", lambda x: update_status(""))
    button0 = tk.Button(text="Custom Model Upload", width=25, anchor="center", command=upload_model,
                        font=("newspaper", 20, "bold"))
    button0.place(relx=0.3, rely=0.65, anchor=tk.CENTER)
    button0.bind("<Enter>", lambda x: update_status("custom"))
    button0.bind("<Leave>", lambda x: update_status(""))

    self_train = tk.IntVar()
    option1 = tk.Checkbutton(text="Self Training", variable=self_train, font=("newspaper", 20, "bold"))
    option1.place(relx=0.8, rely=0.25, anchor=tk.CENTER)
    label5 = tk.Label(text="If you tick the Self Training, \nPlease upload the required files below:"
                           "\n(See Information and Instruction) ", justify="left", font=("newspaper", 12, "bold"), fg="BLACK")
    label5.place(relx=0.8, rely=0.3, anchor=tk.CENTER)

    button1 = tk.Button(text="Build Model", activeforeground="RED", width=25, anchor="center",
                        font=("newspaper", 20, "bold"), command=lambda: turn_page(4))
    button1.place(relx=0.3, rely=0.85, anchor=tk.CENTER)
    button3 = tk.Button(text="Main Menu", activeforeground="RED", width=25, anchor="center",
                        font=("newspaper", 20, "bold"), command=lambda: turn_page(1))
    button3.place(relx=0.7, rely=0.85, anchor=tk.CENTER)
    app.update()


def menu():
    global status_message
    clear_key()
    status_message.set("Status: Menu")
    label1 = tk.Label(text="Machine Learning Monkey Windtex Estimation Regressor", font=("newspaper", 30, "bold"))
    label1.place(relx=0.5, rely=0.25, anchor=tk.CENTER)
    labelv = tk.Label(text="Version " + version, font=("newspaper", 15, "bold"))
    labelv.place(relx=0.9, rely=0.9, anchor=tk.CENTER)
    label2 = tk.Label(text="Please click the button or press the key to continue", font=("newspaper", 20, "bold"))
    label2.place(relx=0.5, rely=0.35, anchor=tk.CENTER)
    button1 = tk.Button(text="1. Instruction and Information", activeforeground="RED", width=35, anchor="w",
                        font=("newspaper", 25, "bold"), command=close)
    button1.place(relx=0.5, rely=0.45, anchor=tk.CENTER)
    button1.bind("<Enter>", lambda x: update_status("b1"))
    button1.bind("<Leave>", lambda x: update_status(""))
    button2 = tk.Button(text="2. Windtex Estimation Regressor", activeforeground="RED", width=35, anchor="w",
                        font=("newspaper", 25, "bold"), command=lambda: turn_page(3))
    button2.place(relx=0.5, rely=0.55, anchor=tk.CENTER)
    button2.bind("<Enter>", lambda x: update_status("b2"))
    button2.bind("<Leave>", lambda x: update_status(""))
    button0 = tk.Button(text="0. Quit", activeforeground="RED", width=35, anchor="w",
                        font=("newspaper", 25, "bold"), command=close)
    button0.place(relx=0.5, rely=0.65, anchor=tk.CENTER)
    button0.bind("<Enter>", lambda x: update_status("quit"))
    button0.bind("<Leave>", lambda x: update_status(""))


def show():
    global page, status_message
    clear()
    label0 = tk.Label(text="", bg="WHITE", justify="center", width=150, height=3, textvariable=status_message,
                      font=("newspaper", 15, "bold"))
    label0.place(relx=0.5, rely=1, anchor="s")
    if page == 0:
        begin()
    elif page == 1:
        menu()
    elif page == 3:
        start()
    elif page == 4:
        build_model()


app = tk.Tk()
app.title("Machine Learning Monkey - Windtex Estimation Regressor")
app.geometry('1200x800')
version = "0.1"
page = 0
status_message = tk.StringVar(app, "Status: Ready")
model_choice = tk.IntVar()
model = None
model_list = {-1: "Custom Model",
                          1: "Ada Boost Regression",
                          2: "Decision Tree Regression",
                          3: "Support Vector Regression",
                          4: "Huber Regression",
                          5: "Logistic Regression",
                          6: "Linear Regression",
                          7: "Kernel Ridge Regression",
                          }
show()
app.mainloop()
