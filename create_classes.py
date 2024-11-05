import os

# Define the base directory for the UNO cards
base_dir = 'c:/Users/ADMIN/Downloads/Test/dataset/images/train'  

# Define the classes for UNO cards
card_classes = [
    'red_0', 'red_1', 'red_2', 'red_3', 'red_4', 'red_5', 'red_6', 'red_7', 'red_8', 'red_9',
    'yellow_0', 'yellow_1', 'yellow_2', 'yellow_3', 'yellow_4', 'yellow_5', 'yellow_6', 'yellow_7', 'yellow_8', 'yellow_9',
    'green_0', 'green_1', 'green_2', 'green_3', 'green_4', 'green_5', 'green_6', 'green_7', 'green_8', 'green_9',
    'blue_0', 'blue_1', 'blue_2', 'blue_3', 'blue_4', 'blue_5', 'blue_6', 'blue_7', 'blue_8', 'blue_9',
    'red_skip', 'red_reverse', 'red_draw_two',
    'yellow_skip', 'yellow_reverse', 'yellow_draw_two',
    'green_skip', 'green_reverse', 'green_draw_two',
    'blue_skip', 'blue_reverse', 'blue_draw_two',
    'wild', 'wild_draw_four'
]

# Create the directories for each class
for card_class in card_classes:
    class_dir = os.path.join(base_dir, card_class)
    os.makedirs(class_dir, exist_ok=True)

print("UNO card classes created successfully!")
