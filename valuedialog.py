# value_dialog.py
import tkinter as tk

class ValueDialog:
    def __init__(self, master, field_names):
        self.master = master
        self.master.title("Enter Values")

        # Set the size of the dialog box
        self.master.geometry("500x500")  # Width x Height

        # Calculate the center position
        self.center_window()

        # Create a frame to hold the labels and entries
        frame = tk.Frame(master)
        frame.pack(expand=True)  # Center the frame in the dialog box

        # Create entry fields and labels based on provided field names
        self.entries = []  # List to store entry references
        for i, field_name in enumerate(field_names):
            tk.Label(frame, text=field_name).grid(row=i, column=0, padx=10, pady=5)
            entry = tk.Entry(frame)
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.entries.append(entry)  # Store the entry reference

        # Submit button
        tk.Button(frame, text="Submit", command=self.submit_values).grid(row=len(field_names), column=0, columnspan=2, pady=10)

        # Variable to store values
        self.values = None

    def center_window(self):
        # Get the dimensions of the screen
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # Calculate x and y coordinates to center the window
        x = (screen_width // 2) - (500 // 2)  # 500 is the width of the window
        y = (screen_height // 2) - (500 // 2)  # 500 is the height of the window

        # Set the position of the window
        self.master.geometry(f"+{x}+{y}")

    def submit_values(self):
        # Retrieve and store values from each entry field
        self.values = [entry.get() for entry in self.entries]
        self.master.destroy()  # Close the dialog


