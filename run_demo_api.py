"""Launch the FastAPI server in demo mode."""
import os
os.environ["DEMO_MODE"] = "1"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
