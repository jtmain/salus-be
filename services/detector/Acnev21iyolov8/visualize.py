import os
import cv2

def visualize_bounding_boxes(image_folder, label_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(root, file)
                label_path = os.path.join(label_folder, f"{os.path.splitext(file)[0]}.txt")

                if os.path.exists(label_path):
                    # Load the image
                    image = cv2.imread(image_path)
                    height, width, _ = image.shape

                    # Load the labels
                    with open(label_path, "r") as f:
                        labels = f.readlines()

                    # Draw bounding boxes
                    for label in labels:
                        class_id, x_center, y_center, bbox_width, bbox_height = map(float, label.strip().split())
                        # Convert normalized coordinates back to absolute pixel values
                        x_center = int(x_center * width)
                        y_center = int(y_center * height)
                        bbox_width = int(bbox_width * width)
                        bbox_height = int(bbox_height * height)

                        # Calculate top-left and bottom-right corners
                        x_min = int(x_center - bbox_width / 2)
                        y_min = int(y_center - bbox_height / 2)
                        x_max = int(x_center + bbox_width / 2)
                        y_max = int(y_center + bbox_height / 2)

                        # Draw the rectangle and class ID
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(image, f"Class {int(class_id)}", (x_min, y_min - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Save the output image
                    relative_path = os.path.relpath(image_path, image_folder)
                    output_image_path = os.path.join(output_folder, relative_path)
                    output_subfolder = os.path.dirname(output_image_path)
                    if not os.path.exists(output_subfolder):
                        os.makedirs(output_subfolder)

                    cv2.imwrite(output_image_path, image)
                    print(f"Bounding boxes visualized and saved: {output_image_path}")

# Paths
image_folder = "./data/train/images"  # Update this path for each dataset split
label_folder = "./data/train/labels"  # Update this path for each dataset split
output_folder = "output_visualized/train"  # Folder to save visualized images

visualize_bounding_boxes(image_folder, label_folder, output_folder)
