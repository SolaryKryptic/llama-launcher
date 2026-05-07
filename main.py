from tkinter import ttk, Tk
import sys
sys.path.insert(0, '.')
from gui_builder import LlamaServerGUI

root = Tk()
gui = LlamaServerGUI(root)
root.title("llama-server CLI Generator")
root.geometry("720x680")
root.mainloop()
