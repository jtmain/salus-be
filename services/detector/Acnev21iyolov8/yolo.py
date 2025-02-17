from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # this means we are building a model from scratch
# Use the model
# train the model
model.train(
data='data.yaml',
epochs=200,     
batch=16,   
imgsz=1280,    
optimizer='AdamW', 
lr0=0.0003,        
lrf=0.02,        
momentum=0.9,   
weight_decay=0.0001,
mosaic=False,       
mixup=False,        
hsv_s=0.3,          
hsv_v=0.3,          
translate=0.05,   
scale=0.9,          
patience=75,        
project='runs/acne_exp_optimized',
name='exp_final',
save_period=10,
verbose=True,

)



