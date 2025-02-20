from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from services.detector.Acnev21iyolov8.run import YOLOImageProcessor
from services.AI.agent import AskTheDoc
import base64 
import uvicorn


BASE_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000")

app = FastAPI()

OUTPUT_FOLDER = os.path.abspath("services/detector/Acnev21iyolov8/output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_FOLDER), name="output")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

processor = YOLOImageProcessor(
    model_path="./services/detector/Acnev21iyolov8/runs/acne_exp_optimized/exp_final/weights/best.pt",
    confidence_threshold=0
)

AI = AskTheDoc()


@app.post("/upload")
async def upload_image(image: UploadFile = File(...), skinfo: str = Form(...)):  
    try:
        filename = image.filename  
        print(f"📢 Received image: {filename}, User Info: {skinfo}")

        skinfo = skinfo.strip() if skinfo.strip() else "No user input provided"

        contents = await image.read()
        image_array = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")

        print("here")
        processed_image, acne_counts = processor.process_image(image)
        print("erere")

        if not acne_counts:
            acne_counts = {"message": "No lesions detected"}
        
        print(f"📊 Acne Count Detected: {acne_counts}")

        output_filename = f"processed_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, processed_image)

        if not os.path.exists(output_path):
            print(f"❌ Image was NOT saved at {output_path}")
        else:
            print(f"✅ Image saved successfully at {output_path}")

        with open(output_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        try:
            print("📡 Sending Base64 Image to OpenRouter...")
            GPT_response = AI.ask(skinfo, acne_counts, output_path)

            if not GPT_response:  
                GPT_response = "No AI response received."
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling OpenRouter: {str(e)}")

        

        with open(output_path, 'rb') as response_image_file:
            image_blob = response_image_file.read()

        encoded_image = base64.b64encode(image_blob).decode('utf-8')

        return JSONResponse(content={
            "message": "Image processed successfully!",
            "acne_counts": acne_counts, 
            "ai_response": GPT_response,
            "response_image": encoded_image
        })

        

    except HTTPException as http_err:    
        return JSONResponse(content={"error": http_err.detail}, status_code=http_err.status_code)

    except Exception as e:
        print(f"❌ General Error: {str(e)}")
        return JSONResponse(content={"error": "An unexpected error occurred."}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
