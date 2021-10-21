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


def build_model():
    global status_message, page, model_choice, model, self_train, trained_model
    data = None
    m_choice = model_choice.get()
    s_train = self_train.get()
    label1 = tk.Label(text="Machine Learning Monkey Windtex Estimation Regressor", font=("newspaper", 30, "bold"))
    label1.place(relx=0.5, rely=0.05, anchor=tk.CENTER)

    label2 = tk.Label(text="ML Monkey Model Setting Done!\nSelected Model: " + model_list[m_choice],
                      font=("newspaper", 25, "bold"))
    label2.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label3 = tk.Label(text="Feature Extraction", font=("newspaper", 20, "bold"), fg="BLACK")
    label3.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
    label4 = tk.Label(text="Model Initialisation", font=("newspaper", 20, "bold"), fg="BLACK")
    label4.place(relx=0.5, rely=0.35, anchor=tk.CENTER)
    label5 = tk.Label(text="Model Training", font=("newspaper", 20, "bold"), fg="BLACK")
    label5.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    label6 = tk.Label(text="Model Validation", font=("newspaper", 20, "bold"), fg="BLACK")
    label6.place(relx=0.5, rely=0.45, anchor=tk.CENTER)

    if s_train != 1:
        self_file["data"] = "./OriginalData/Calculator.csv"
        self_file["image"] = "./OriginalData/data/"
        self_file["label"] = "./OriginalData/label_2.json"
    valid = {}
    if m_choice != -1 and s_train == 0:
        with open("./TrainedModels/" + model_list[m_choice] + ".pkl", "rb") as pk:
            model = pickle.load(pk)
            pk.close()
        if model_list[m_choice] == "Ada Boost Regression":
            valid = {"self": {"MAE": 0, "ACC-0": 1, "ACC-1": 1, "ACC-2": 1, "ACC-3": 1},
                     "LOO": {"MAE": 0.74, "ACC-0": 0.496, "ACC-1": 0.854, "ACC-2": 0.946, "ACC-3": 0.982},
                     "RV": {"MAE": 0.71, "ACC-0": 0.497, "ACC-1": 0.861, "ACC-2": 0.96, "ACC-3": 0.985}}
        elif model_list[m_choice] == "Logistic Regression":
            valid = {"self": {"MAE": 0.64, "ACC-0": 0.598, "ACC-1": 0.848, "ACC-2": 0.942, "ACC-3": 0.976},
                     "LOO": {"MAE": 0.96, "ACC-0": 0.446, "ACC-1": 0.768, "ACC-2": 0.898, "ACC-3": 0.954},
                     "RV": {"MAE": 0.96, "ACC-0": 0.421, "ACC-1": 0.76, "ACC-2": 0.91, "ACC-3": 0.967}}
        elif model_list[m_choice] == "Huber Regression":
            valid = {"self": {"MAE": 0.9, "ACC-0": 0.36, "ACC-1": 0.83, "ACC-2": 0.948, "ACC-3": 0.976},
                     "LOO": {"MAE": 0.98, "ACC-0": 0.31, "ACC-1": 0.802, "ACC-2": 0.948, "ACC-3": 0.968},
                     "RV": {"MAE": 0.99, "ACC-0": 0.306, "ACC-1": 0.801, "ACC-2": 0.944, "ACC-3": 0.969}}
        elif model_list[m_choice] == "Linear Regression":
            valid = {"self": {"MAE": 0.88, "ACC-0": 0.354, "ACC-1": 0.82, "ACC-2": 0.954, "ACC-3": 0.992},
                     "LOO": {"MAE": 0.99, "ACC-0": 0.33, "ACC-1": 0.778, "ACC-2": 0.934, "ACC-3": 0.978},
                     "RV": {"MAE": 0.99, "ACC-0": 0.324, "ACC-1": 0.782, "ACC-2": 0.933, "ACC-3": 0.979}}
        elif model_list[m_choice] == "Decision Tree Regression":
            valid = {"self": {"MAE": 0, "ACC-0": 1, "ACC-1": 1, "ACC-2": 1, "ACC-3": 1},
                     "LOO": {"MAE": 1.01, "ACC-0": 0.436, "ACC-1": 0.758, "ACC-2": 0.888, "ACC-3": 0.948},
                     "RV": {"MAE": 1.06, "ACC-0": 0.399, "ACC-1": 0.735, "ACC-2": 0.885, "ACC-3": 0.947}}
        elif model_list[m_choice] == "Support Vector Regression":
            valid = {"self": {"MAE": 0.42, "ACC-0": 0.704, "ACC-1": 0.914, "ACC-2": 0.372, "ACC-3": 0.365},
                     "LOO": {"MAE": 0.95, "ACC-0": 0.372, "ACC-1": 0.798, "ACC-2": 0.928, "ACC-3": 0.968},
                     "RV": {"MAE": 1.02, "ACC-0": 0.365, "ACC-1": 0.805, "ACC-2": 0.932, "ACC-3": 0.97}}
        elif model_list[m_choice] == "Ridge Regression":
            valid = {"self": {"MAE": 0.89, "ACC-0": 0.356, "ACC-1": 0.822, "ACC-2": 0.952, "ACC-3": 0.986},
                     "LOO": {"MAE": 0.97, "ACC-0": 0.334, "ACC-1": 0.786, "ACC-2": 0.938, "ACC-3": 0.98},
                     "RV": {"MAE": 0.97, "ACC-0": 0.334, "ACC-1": 0.777, "ACC-2": 0.938, "ACC-3": 0.985}}

    elif m_choice != -1 and s_train == 1:
        if model_list[m_choice] == "Ada Boost Regression":
            from sklearn.ensemble import AdaBoostRegressor
            from sklearn.tree import DecisionTreeRegressor
            model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(criterion="mse"), learning_rate=0.00001,
                                      n_estimators=200)

        elif model_list[m_choice] == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(solver="liblinear", max_iter=5000)

        elif model_list[m_choice] == "Huber Regression":
            from sklearn.linear_model import HuberRegressor
            model = HuberRegressor(max_iter=2000, epsilon=1.5)

        elif model_list[m_choice] == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

        elif model_list[m_choice] == "Decision Tree Regression":
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor(criterion="mse")

        elif model_list[m_choice] == "Support Vector Regression":
            from sklearn.svm import SVR
            model = SVR(kernel="poly", degree=4, coef0=4, epsilon=0.1)

        elif model_list[m_choice] == "Ridge Regression":
            from sklearn.linear_model import Ridge
            model = Ridge(solver="saga")

    if model is None:
        messagebox.showerror("No model loaded", "Error: No model loaded\nPlease upload again!")
        status_message.set("Uploaded model is not found! Please upload again!")
        app.update()
        turn_page(3)

    if m_choice == -1 or s_train == 1:
        label3["fg"] = "BLUE"
        status_message.set("Feature Extracting... Please wait...")
        app.update()
        data = Windtex.prepare(self_file["data"], self_file["image"], self_file["label"])
        status_message.set("Feature Extraction Done!\n"
                           "Initialising Model... Please wait...")
        app.update()
        label3["fg"] = "GREEN"
        label4["fg"] = "BLUE"
        WindtexModel.born(data)
        label4["fg"] = "GREEN"
        label5["fg"] = "BLUE"
        status_message.set("Model Initialised!\n"
                           "Training Model... Please wait...")
        app.update()
        trained, trained_ex, valid = WindtexModel.grow(model=model)
        label5["fg"] = "GREEN"
        label6["fg"] = "BLUE"
        status_message.set("Model Trained!\n"
                           "Validating Model... Please wait...")
        app.update()
        valid_result = "\t\t\tMSE\tRMSE\tACC-0\tACC-0.5\tACC-1\tACC-1.5\n"
        for i in valid:
            if i == "self":
                valid_result += "Self Validation:\t"
            elif i == "LOO":
                valid_result += "Leave One Out:\t\t"
            elif i == "RV":
                valid_result += "Random Validation:\t"
            valid_result += "\t".join([str(round(i, 2)) for i in [valid[i]["MAE"], valid[i]["RMSE"],
                                                                  valid[i]["ACC-0"] * 100,
                                                                  valid[i]["ACC-1"] * 100,
                                                                  valid[i]["ACC-2"] * 100,
                                                                  valid[i]["ACC-3"] * 100]]) + "\n"

        label_vr = tk.Label(text=valid_result, font=("newspaper", 15, "bold"), fg="BLACK", justify="left")
        label_vr.place(relx=0.5, rely=0.55, anchor=tk.CENTER)
        label6["fg"] = "GREEN"
        trained_model = trained
        status_message.set("Model Trained and Loaded!\n"
                           "Please view the validation result, and continue...")
        app.update()
    else:
        label3["fg"] = "GREEN"
        label4["fg"] = "GREEN"
        label5["fg"] = "GREEN"
        label6["fg"] = "GREEN"
        valid_result = "\t\t\tMSE\tACC-0\tACC-0.5\tACC-1\tACC-1.5\n"
        for i in valid:
            if i == "self":
                valid_result += "Self Validation:\t"
            elif i == "LOO":
                valid_result += "Leave One Out:\t\t"
            elif i == "RV":
                valid_result += "Random Validation:\t"
            valid_result += "\t".join([str(round(i, 2)) for i in [valid[i]["MAE"],
                                                                  valid[i]["ACC-0"] * 100,
                                                                  valid[i]["ACC-1"] * 100,
                                                                  valid[i]["ACC-2"] * 100,
                                                                  valid[i]["ACC-3"] * 100]]) + "\n"
        label_vr = tk.Label(text=valid_result, font=("newspaper", 15, "bold"), fg="BLACK", justify="left")
        label_vr.place(relx=0.5, rely=0.55, anchor=tk.CENTER)
        trained_model = model
        status_message.set("Model Loaded!\n"
                           "Please view the validation result, and continue...")
        app.update()

    if trained_model is not None:
        button9 = tk.Button(text="Continue", activeforeground="RED", width=25, anchor="center",
                            font=("newspaper", 20, "bold"), command=lambda: turn_page(5))
        button9.place(relx=0.5, rely=0.75, anchor=tk.CENTER)

    button3 = tk.Button(text="Main Menu", activeforeground="RED", width=25, anchor="center",
                        font=("newspaper", 20, "bold"), command=lambda: turn_page(1))
    button3.place(relx=0.3, rely=0.85, anchor=tk.CENTER)
    button2 = tk.Button(text="Back", activeforeground="RED", width=25, anchor="center",
                        font=("newspaper", 20, "bold"), command=lambda: turn_page(3))
    button2.place(relx=0.7, rely=0.85, anchor=tk.CENTER)


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
        status_message.set('Ridge(solver="saga")'
                           '\nMean Absolute Error: 0.95-0.96')
    elif button == "custom":
        status_message.set('Please store the algorithm model into a pickle file and upload here')
    elif button == "data_up":
        status_message.set('Please upload the Data CSV file here (content refer to instruction page)')
    elif button == "img_up":
        status_message.set('Please upload the image folder here (refer to instruction page)')
    elif button == "label_up":
        status_message.set('Please upload the Label JSON here (content refer to instruction page)')
    else:
        if page == 1:
            status_message.set("Status: Menu")
        elif page == 3:
            status_message.set("Status: ML Monkey Setting\nSelected Model: " + model_list[model_choice.get()])
        elif page == 4:
            status_message.set("Status: ML Monkey Loading/Training")
        elif page == 5:
            status_message.set("Status: ML Monkey Predicting")


def start():
    global status_message, model, model_choice, self_train, trained_model

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
            label_model = tk.Label(text="Model Uploaded", font=("newspaper", 10, "bold"), fg="BLUE")
            label_model.place(relx=0.3, rely=0.7, anchor=tk.CENTER)

    def upload_self_data(which):
        global self_file
        from tkinter import filedialog
        self_path = None
        if which == "data":
            self_path = filedialog.askopenfilename(filetypes=[("CSV files", ".csv .CSV")])
        elif which == "label":
            self_path = filedialog.askopenfilename(filetypes=[("JSON files", ".json .JSON")])
        elif which == "image":
            self_path = filedialog.askdirectory()
        self_file[which] = os.path.abspath(self_path)
        if self_file["data"] is not None:
            label_data = tk.Label(text="Data CSV Uploaded", font=("newspaper", 10, "bold"), fg="BLUE")
            label_data.place(relx=0.8, rely=0.4, anchor=tk.CENTER)
        if self_file["image"] is not None:
            label_img = tk.Label(text="Image Folder Uploaded", font=("newspaper", 10, "bold"), fg="BLUE")
            label_img.place(relx=0.8, rely=0.5, anchor=tk.CENTER)
        if self_file["label"] is not None:
            label_le = tk.Label(text="Label JSON Uploaded", font=("newspaper", 10, "bold"), fg="BLUE")
            label_le.place(relx=0.8, rely=0.6, anchor=tk.CENTER)

    clear_key()

    label1 = tk.Label(text="Machine Learning Monkey Windtex Estimation Regressor", font=("newspaper", 30, "bold"))
    label1.place(relx=0.5, rely=0.05, anchor=tk.CENTER)
    label2 = tk.Label(text='Directly click the "Build Model" to execute the optimal settings\nor follow the step below', font=("newspaper", 20, "bold"))
    label2.place(relx=0.5, rely=0.12, anchor=tk.CENTER)
    label3 = tk.Label(text="Step 1: Preparing the model", font=("newspaper", 25, "bold"), fg="RED")
    label3.place(relx=0.3, rely=0.2, anchor=tk.CENTER)
    label4 = tk.Label(text="Step 2: ML Monkey Options", font=("newspaper", 25, "bold"), fg="RED")
    label4.place(relx=0.8, rely=0.2, anchor=tk.CENTER)

    check1 = tk.Radiobutton(text="1. Ada Boost Regression - Decision Tree Regression", value=1,
                            font=("newspaper", 15, "bold"), var=model_choice)
    check1.place(relx=0.3, rely=0.25, anchor=tk.CENTER)
    check1.bind("<Enter>", lambda x: update_status("adaboost"))
    check1.bind("<Leave>", lambda x: update_status(""))
    check2 = tk.Radiobutton(text="2. Decision Tree Regression", value=2, var=model_choice,
                            font=("newspaper", 15, "bold"))
    check2.place(relx=0.3, rely=0.3, anchor=tk.CENTER)
    check2.bind("<Enter>", lambda x: update_status("dt"))
    check2.bind("<Leave>", lambda x: update_status(""))
    check3 = tk.Radiobutton(text="3. Support Vector Regression", value=3, var=model_choice,
                            font=("newspaper", 15, "bold"))
    check3.place(relx=0.3, rely=0.35, anchor=tk.CENTER)
    check3.bind("<Enter>", lambda x: update_status("svr"))
    check3.bind("<Leave>", lambda x: update_status(""))
    check4 = tk.Radiobutton(text="4. Huber Regression", value=4, var=model_choice, font=("newspaper", 15, "bold"))
    check4.place(relx=0.3, rely=0.4, anchor=tk.CENTER)
    check4.bind("<Enter>", lambda x: update_status("huber"))
    check4.bind("<Leave>", lambda x: update_status(""))
    check5 = tk.Radiobutton(text="5. Logistic Regression", value=5, var=model_choice, font=("newspaper", 15, "bold"))
    check5.place(relx=0.3, rely=0.45, anchor=tk.CENTER)
    check5.bind("<Enter>", lambda x: update_status("log"))
    check5.bind("<Leave>", lambda x: update_status(""))
    check6 = tk.Radiobutton(text="6. Linear Regression", value=6, var=model_choice, font=("newspaper", 15, "bold"))
    check6.place(relx=0.3, rely=0.5, anchor=tk.CENTER)
    check6.bind("<Enter>", lambda x: update_status("linear"))
    check6.bind("<Leave>", lambda x: update_status(""))
    check7 = tk.Radiobutton(text="7. Ridge Regression", value=7, var=model_choice, font=("newspaper", 15, "bold"))
    check7.place(relx=0.3, rely=0.55, anchor=tk.CENTER)
    check7.bind("<Enter>", lambda x: update_status("kridge"))
    check7.bind("<Leave>", lambda x: update_status(""))
    check0 = tk.Radiobutton(text="Custom Model (required fit() and predict() function)", value=-1, var=model_choice,
                            font=("newspaper", 15, "bold"))
    check0.place(relx=0.3, rely=0.6, anchor=tk.CENTER)
    check0.bind("<Enter>", lambda x: update_status("custom"))
    check0.bind("<Leave>", lambda x: update_status(""))
    button0 = tk.Button(text="Custom Model Upload", width=25, anchor="center", command=upload_model,
                        font=("newspaper", 15, "bold"))
    button0.place(relx=0.3, rely=0.65, anchor=tk.CENTER)
    button0.bind("<Enter>", lambda x: update_status("custom"))
    button0.bind("<Leave>", lambda x: update_status(""))

    option1 = tk.Checkbutton(text="Self Training", variable=self_train, font=("newspaper", 15, "bold"))
    option1.place(relx=0.8, rely=0.25, anchor=tk.CENTER)
    label5 = tk.Label(text="If you tick the Self Training, \nPlease upload the required files below:"
                           "\n(See Information and Instruction) ", justify="left", font=("newspaper", 10, "bold"),
                      fg="BLACK")
    label5.place(relx=0.8, rely=0.3, anchor=tk.CENTER)
    button4 = tk.Button(text="Data CSV File Upload", width=25, anchor="center",
                        command=lambda: upload_self_data("data"),
                        font=("newspaper", 15, "bold"))
    button4.place(relx=0.8, rely=0.35, anchor=tk.CENTER)
    button4.bind("<Enter>", lambda x: update_status("csv_up"))
    button4.bind("<Leave>", lambda x: update_status(""))
    button5 = tk.Button(text="Image Folder Upload", width=25, anchor="center",
                        command=lambda: upload_self_data("image"),
                        font=("newspaper", 15, "bold"))
    button5.place(relx=0.8, rely=0.45, anchor=tk.CENTER)
    button5.bind("<Enter>", lambda x: update_status("img_up"))
    button5.bind("<Leave>", lambda x: update_status(""))
    button6 = tk.Button(text="Label JSon File Upload", width=25, anchor="center",
                        command=lambda: upload_self_data("label"),
                        font=("newspaper", 15, "bold"))
    button6.place(relx=0.8, rely=0.55, anchor=tk.CENTER)
    button6.bind("<Enter>", lambda x: update_status("label_up"))
    button6.bind("<Leave>", lambda x: update_status(""))

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


def execute_test():
    global status_message, page, model_choice, model, self_train, trained_model, test_folder
    test_folder = str(time.strftime("%Y%m%d%H%M%S"))
    if os.path.exists("./test"):
        os.mkdir('./.test')

    os.mkdir("./.test/" + test_folder)
    os.mkdir("./.test/" + test_folder + "/image")

    def upload_img():
        from tkinter import filedialog
        from shutil import copy
        image_paths = filedialog.askopenfilenames(filetypes=[("Image Files", ".jpg .JPG .jpeg .JPEG .png .PNG")])
        id_count = 0
        for i in image_paths:
            copy(i, "./.test/" + test_folder + "/image/"+str(id_count)+".jpg")
            id_count += 1
        label3["fg"] = "GREEN"

    def label_img():
        if label3["fg"] != "GREEN":
            messagebox.showerror("No Images Uploaded", "No Images are uploaded!\nPlease upload images first")
            pass
        reply = messagebox.askokcancel("Turning to VIA tool", "VIA Tools will be opened after you clicking OK button\n"
                                                      "Please annotate the defect areas by using polygons\n"
                                                      "Please press save icon after the annotation")

        if reply:
            import webbrowser
            webbrowser.open("./via.html")

    label1 = tk.Label(text="Machine Learning Monkey Windtex Estimation Regressor", font=("newspaper", 30, "bold"))
    label1.place(relx=0.5, rely=0.05, anchor=tk.CENTER)
    label2 = tk.Label(text='Please follow the steps below to execute prediction',
                      font=("newspaper", 20, "bold"))
    label2.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    label3 = tk.Label(text="Step 1: Upload Test Images", font=("newspaper", 15, "bold"), fg="RED")
    label3.place(relx=0.3, rely=0.2, anchor=tk.CENTER)
    button3 = tk.Button(text="Upload Images", width=25, anchor="center", command=upload_img,
                        font=("newspaper", 15, "bold"))
    button3.place(relx=0.6, rely=0.2, anchor=tk.CENTER)
    label4 = tk.Label(text="Step 2: Label Test Images", font=("newspaper", 15, "bold"), fg="RED")
    label4.place(relx=0.3, rely=0.25, anchor=tk.CENTER)
    button4 = tk.Button(text="Label Images", width=25, anchor="center", command=label_img,
                        font=("newspaper", 15, "bold"))
    button4.place(relx=0.6, rely=0.25, anchor=tk.CENTER)


def show():
    global page, status_message
    clear()
    label0 = tk.Label(text="", bg="WHITE", justify="center", width=150, height=3, textvariable=status_message,
                      font=("newspaper", 15, "bold"))
    label0.place(relx=0.5, rely=1, anchor="s")
    update_status("")
    if page == 0:
        begin()
    elif page == 1:
        menu()
    elif page == 3:
        start()
    elif page == 4:
        build_model()
    elif page == 5:
        execute_test()


app = tk.Tk()
app.title("Machine Learning Monkey - Windtex Estimation Regressor")
app.geometry('1200x800')
version = "0.1"
page = 0
status_message = tk.StringVar(app, "Status: Ready")
model_choice = tk.IntVar()
self_train = tk.IntVar()
model_choice.set(1)
self_file = {"image": None, "data": None, "label": None}
model = None
trained_model = None
test_folder = ""
model_list = {-1: "Custom Model",
              1: "Ada Boost Regression",
              2: "Decision Tree Regression",
              3: "Support Vector Regression",
              4: "Huber Regression",
              5: "Logistic Regression",
              6: "Linear Regression",
              7: "Ridge Regression",
              }
show()
app.mainloop()
