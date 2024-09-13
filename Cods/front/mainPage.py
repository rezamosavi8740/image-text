from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import uvicorn
import random

class FastAPIApp:
    def __init__(self):
        self.app = FastAPI()
        self.templates = Jinja2Templates(directory="templates")
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.setup_routes()

        # To store the image path and links between requests
        self.image_path = None
        self.links = []  # List of generated links

    def generate_links(self):
        """Generates random links for demonstration."""
        base_url = "https://example.com/link"
        self.links = [f"{base_url}/{random.randint(1000, 9999)}" for _ in range(3)]

    def setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request, "image_path": self.image_path})

        @self.app.post("/upload_image", response_class=HTMLResponse)
        async def upload_image(request: Request, file: UploadFile = File(...)):
            file_location = f"static/{file.filename}"
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Save the image path and generate new links
            self.image_path = file_location
            self.generate_links()

            return self.templates.TemplateResponse("index.html", {
                "request": request,
                "image_path": self.image_path,
                "links": self.links
            })

        @self.app.post("/process_text", response_class=HTMLResponse)
        async def process_text(request: Request, input_field: str = Form(...)):
            result = self.some_python_function(input_field)

            # Generate new links
            self.generate_links()

            return self.templates.TemplateResponse("index.html", {
                "request": request,
                "result": result,
                "image_path": self.image_path,
                "links": self.links
            })

    def some_python_function(self, input_value: str) -> str:
        return f"Processed result: {input_value}"

    def run(self):
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
