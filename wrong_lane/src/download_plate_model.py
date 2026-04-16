import requests
import os

def download_model():
    # Trying an alternative public model
    url = "https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/main/best.pt"
    save_path = "yolov8n_plate.pt"
    
    if os.path.exists(save_path):
        print(f"Model {save_path} already exists.")
        return
        
    print(f"Downloading plate detector model from {url}...")
    try:
        # User-Agent might be needed for some public downloads
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Trying another one if first fails
        url2 = "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"
        print(f"Trying alternative: {url2}")
        try:
            response = requests.get(url2, headers=headers, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete (alternative).")
        except Exception as e2:
            print(f"Error downloading alternative model: {e2}")

if __name__ == "__main__":
    download_model()
