import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse

import uvicorn


# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

with open('users.json') as f:
    users = json.load(f)
users = users["users"]
print(users)

HD_image_path = "HD_image.png"
HD_image_PDF_path = "HD_image_PDF.pdf"

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello!"}

@app.get("/get_hello_world_json")
async def get_hello_world():
    return {"message": "Hello World"}

@app.get("/get_html_response", response_class=HTMLResponse)
async def get_html_response():
    return """
    <html>
        <head>
            <title>Simple HTML response</title>
        </head>
        <body>
            <h1>HTML response is received</h1>
        </body>
    </html>
    """

@app.get("/display_HD_image")
def display_HD_image():
    def iterfile():
        with open(HD_image_path, mode="rb") as file_like:  # Update the path to your HD image
            yield from file_like
    return StreamingResponse(iterfile(), media_type="image/png")

@app.get("/download_HD_image")
def download_HD_image():
    image_path = HD_image_path  # Replace with the path to your PDF file
    return FileResponse(image_path, media_type='application/image', filename="HD_image.png")

@app.get("/download_HD_image_pdf")
def download_HD_image_PDF():
    pdf_file_path = HD_image_PDF_path  # Replace with the path to your PDF file
    return FileResponse(pdf_file_path, media_type='application/pdf', filename="HD_image_pdf.pdf")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
