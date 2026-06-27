from dotenv import load_dotenv
from .server import app
import uvicorn
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

def main():
    uvicorn.run(app, port=8000, host="0.0.0.0")

main()