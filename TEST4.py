# main.py
from fastapi import FastAPI, UploadFile, File
import shutil
import os
from test2 import process_video

app = FastAPI()


@app.post("/process/")
async def process_video_file(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    clip_name, _ = os.path.splitext(file.filename)
    process_video(input_path, clip_name, output_folder)

    os.remove(input_path)  # 删除临时文件

    return {"message": f"Video processed and saved to {output_folder}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)