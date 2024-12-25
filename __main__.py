import uvicorn

from app.server import client

if __name__ == "__main__":
    uvicorn.run(client, host="0.0.0.0", port=8000)
