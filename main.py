from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import os
import shutil
from services.detector.Acnev21iyolov8.run import YOLOImageProcessor
from services.AI.agent import AskTheDoc

app = FastAPI()

OUTPUT_FOLDER = os.path.abspath("services/detector/Acnev21iyolov8/output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

processor = YOLOImageProcessor(
    model_path="./services/detector/Acnev21iyolov8/runs/acne_exp_optimized/exp_final/weights/best.pt",
    output_folder=OUTPUT_FOLDER,
    confidence_threshold=0
)

AI = AskTheDoc(api_key="key")

@app.post("/upload")
async def upload_image(image: UploadFile = File(...), skinfo: str = Form(...)):  
    try:
       
        contents = await image.read() 

        image_array = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")

        processed_image, acne_counts = processor.process_image(image)

        output_filename = f"processed_{image.filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, processed_image)

        try:
            GPT_response = AI.ask(skinfo, acne_counts, output_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling OpenAI: {str(e)}")

        return JSONResponse(content={
            "message": "Image processed successfully!",
            "acne_counts": acne_counts, 
            "ai_response": GPT_response,
            "annotated_image_url": f"http://127.0.0.1:8000/output/{output_filename}"
        })

    except HTTPException as http_err:    
        return JSONResponse(content={"error": http_err.detail}, status_code=http_err.status_code)


    except Exception as e:
        print(f" General Error: {str(e)}")
        return JSONResponse(content={"error": "An unexpected error occurred."}, status_code=500)
        

@app.get("/")
def get():
       
        return 

