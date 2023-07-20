"""
Add your object hear with label number.
If you want to add the additional label just add 1 to the previous label.
Keep the same sequence in yolo v8 config file
e.g. class_map = {'gun': '0', 'bottle': '1'}
"""

class_map = {'gun': '0', 'bottle': '1', 'glass': '2'}

# python main.py --folder_name videos --object_name cigarette --classes_name smoking --BOX_THRESHOLD 0.3
