import tkinter as tk
from tkinter import scrolledtext

# Create the main window
root = tk.Tk()
root.title("Spam Detection Project")
root.geometry("1080x800")  # Set the window size

# Add a big heading label
heading_label = tk.Label(root, text="Spam Detection Project", font=("Helvetica", 36, "bold"))
heading_label.pack(pady=20)

# Create a large text area with custom styling
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10, width=50, font=("Helvetica", 14))
text_area.pack(padx=20, pady=20)

# Create an "Enter" button with a colorful background
enter_button = tk.Button(root, text="Enter", bg="#FF5733", fg="white", font=("Helvetica", 16))
enter_button.pack()

# Create a label to display the captured input
input_label = tk.Label(root, font=("Helvetica", 14, "italic"), fg="#333333")
input_label.pack()

# Capture input when the button is clicked
def capture_input():
    user_input = text_area.get("1.0", "end-1c")  # Get input from the text area
    input_label.config(text=f"Captured input: {user_input}")  # Update the label text
    print(f"User input: {user_input}")
    print(type(user_input))

enter_button.config(command=capture_input)  # Attach the function to the button

root.mainloop()
