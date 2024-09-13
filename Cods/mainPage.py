from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import RedirectResponse
from pathlib import Path
from Cods.preprossing.ImageEmbedding import ImageEmbeddingPipeline
import random

class WebApp:
    def __init__(self):
        # Initialize the FastAPI app
        self.app = FastAPI()

        # Set up static directory to serve uploaded images and static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

        # Jinja2 templates directory
        self.templates = Jinja2Templates(directory="templates")

        # Directory to store uploaded images
        self.UPLOAD_DIR = "static/uploads"
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

        # Store image and text for persistence across requests
        self.current_image_path = None
        self.current_text = None

        # Links to display (initial)
        self.links = ["https://fastapi.tiangolo.com", "https://python.org"]

        # Set up routes
        self.setup_routes()

    def generate_random_links(self):
        """Generate a random set of links for demo purposes."""
        random_links = [
            f"https://example.com/random_link_{random.randint(1, 100)}",
            f"https://anotherexample.com/unique_{random.randint(1, 100)}"
        ]
        return random_links

    def setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self.templates.TemplateResponse("index.html", {
                "request": request,
                "image_path": self.current_image_path,
                "result": self.current_text,
                "links": self.links
            })

        @self.app.post("/upload_image")
        async def upload_image(file: UploadFile = File(...)):
            # Save the uploaded image to the uploads directory
            file_location = f"{self.UPLOAD_DIR}/{file.filename}"
            with open(file_location, "wb") as f:
                f.write(await file.read())

            # Store the path for rendering
            print(file.filename)
            self.current_image_path = f"/static/uploads/{file.filename}"
            listnames = []
            listnames.append(file.filename)
            print(f"address : {self.current_image_path}")
            self.myImageEmbdding = ImageEmbeddingPipeline(listnames, "/static/uploads/")
            results = self.myImageEmbdding.run()
            print(results)
            #print(results[file.filename])


            # Update links randomly when an image is uploaded
            self.links = self.generate_random_links()

            return RedirectResponse(url="/", status_code=303)

        @self.app.post("/process_text")
        async def process_text(input_field: str = Form(...)):
            # Store the processed text
            self.current_text = input_field

            # Update links randomly when text is submitted
            self.links = self.generate_random_links()

            return RedirectResponse(url="/", status_code=303)

