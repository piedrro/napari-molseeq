import os
import xml.etree.ElementTree as ET

def parse_ui_file(ui_file_path):

    tree = ET.parse(ui_path)
    root = tree.getroot()

    widgets = root.findall(".//widget")

    gui_elements = {}

    for widget in widgets:
        # Get the type of widget and its name property
        widget_type = widget.get('class')
        name = widget.get('name')

        if widget_type in ["QComboBox", "QCheckBox", "QLineEdit","QDoubleSpinBox",
                           "QSpinBox", "QSlider", "QLabel", "QPushButton","QProgressBar"]:

            gui_elements[name] = widget_type

    return gui_elements

ui_path = os.path.join(os.path.dirname(__file__), "GUI", "pixseq_ui.ui")
gui_elements = parse_ui_file(ui_path)






directory = r"C:\Users\turnerp\PycharmProjects\napari-PixSeq\src\napari_pixseq"


python_files = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            python_files.append(path)

for path in python_files:

    with open(path, "r") as f:
        lines = f.readlines()

    print(lines)


    break



